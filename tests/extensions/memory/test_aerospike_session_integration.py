"""Integration tests for AerospikeSession against a real Aerospike CE server.

Unlike test_aerospike_session.py, nothing here is mocked: these tests exercise
the real wire protocol via the ``aerospike`` client. The whole module is
skipped when no server is reachable, so contributors without a container
running still get a green default test run.

Start a server with::

    docker run -d --ulimit nofile=20000:20000 -p 3000:3000 \\
        -e "NAMESPACE=test" aerospike/aerospike-server:latest

See AERO_VALIDATION.md at the repo root for a full manual walkthrough.
"""

from __future__ import annotations

import os
import socket
import sys
import uuid
from collections.abc import Iterator

import pytest

pytest.importorskip("aerospike")

import aerospike  # noqa: E402

# test_aerospike_session.py imports the same production module against a
# fake aerospike package for its unit-test tier. Whichever test file runs
# first in this pytest session leaves its own bound copy of
# agents.extensions.memory.aerospike_session cached in sys.modules; without
# eviction here, this file could silently reuse that stale (fake-bound) copy
# instead of importing fresh against the real client just imported above.
sys.modules.pop("agents.extensions.memory.aerospike_session", None)
_parent_memory_package = sys.modules.get("agents.extensions.memory")
if _parent_memory_package is not None:
    _parent_memory_package.__dict__.pop("aerospike_session", None)

from agents.extensions.memory.aerospike_session import AerospikeSession  # noqa: E402

AEROSPIKE_HOST = os.environ.get("AEROSPIKE_TEST_HOST", "127.0.0.1")
AEROSPIKE_PORT = int(os.environ.get("AEROSPIKE_TEST_PORT", "3000"))
AEROSPIKE_NAMESPACE = os.environ.get("AEROSPIKE_TEST_NAMESPACE", "test")
_SET_NAME = "agent_sessions_it"


def _server_available() -> bool:
    try:
        with socket.create_connection((AEROSPIKE_HOST, AEROSPIKE_PORT), timeout=1):
            return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.aerospike_integration,
    pytest.mark.skipif(
        not _server_available(),
        reason=(
            f"No Aerospike server reachable at {AEROSPIKE_HOST}:{AEROSPIKE_PORT}. Start one "
            'with: docker run -d --ulimit nofile=20000:20000 -p 3000:3000 -e "NAMESPACE=test" '
            "aerospike/aerospike-server:latest"
        ),
    ),
]


@pytest.fixture(scope="module")
def client() -> Iterator[aerospike.Client]:
    c = aerospike.client({"hosts": [(AEROSPIKE_HOST, AEROSPIKE_PORT)]}).connect()
    yield c
    c.close()


@pytest.fixture
def session_id() -> str:
    return f"it-{uuid.uuid4().hex}"


def _make_session(client: aerospike.Client, session_id: str, **kwargs) -> AerospikeSession:
    return AerospikeSession(
        session_id,
        client=client,
        namespace=AEROSPIKE_NAMESPACE,
        set_name=_SET_NAME,
        **kwargs,
    )


async def test_add_and_get_items_roundtrip(client: aerospike.Client, session_id: str) -> None:
    session = _make_session(client, session_id)
    try:
        await session.add_items(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        )
        items = await session.get_items()
        assert [i.get("content") for i in items] == ["Hello", "Hi there!"]
    finally:
        await session.clear_session()


async def test_get_items_empty_session_returns_empty_list(
    client: aerospike.Client, session_id: str
) -> None:
    session = _make_session(client, session_id)
    assert await session.get_items() == []


async def test_get_items_tail_limit_uses_real_negative_index_range(
    client: aerospike.Client, session_id: str
) -> None:
    session = _make_session(client, session_id)
    try:
        for i in range(5):
            await session.add_items([{"role": "user", "content": str(i)}])
        items = await session.get_items(limit=2)
        assert [i.get("content") for i in items] == ["3", "4"]

        # A limit larger than the stored history must clamp, not error.
        items = await session.get_items(limit=100)
        assert len(items) == 5
    finally:
        await session.clear_session()


async def test_pop_item_removes_and_returns_last(client: aerospike.Client, session_id: str) -> None:
    session = _make_session(client, session_id)
    try:
        await session.add_items(
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
            ]
        )
        popped = await session.pop_item()
        assert popped is not None and popped.get("content") == "second"
        assert [i.get("content") for i in await session.get_items()] == ["first"]
    finally:
        await session.clear_session()


async def test_pop_item_on_missing_record_returns_none(
    client: aerospike.Client, session_id: str
) -> None:
    session = _make_session(client, session_id)
    assert await session.pop_item() is None


async def test_pop_item_on_emptied_list_returns_none(
    client: aerospike.Client, session_id: str
) -> None:
    session = _make_session(client, session_id)
    await session.add_items([{"role": "user", "content": "x"}])
    await session.pop_item()
    # The record now exists with an empty list bin — exercises the real
    # AEROSPIKE_ERR_OP_NOT_APPLICABLE path, distinct from a missing record.
    assert await session.pop_item() is None


async def test_clear_session_on_missing_record_is_noop(
    client: aerospike.Client, session_id: str
) -> None:
    session = _make_session(client, session_id)
    await session.clear_session()
    assert await session.get_items() == []


async def test_clear_session_then_reuse(client: aerospike.Client, session_id: str) -> None:
    session = _make_session(client, session_id)
    try:
        await session.add_items([{"role": "user", "content": "gone"}])
        await session.clear_session()
        assert await session.get_items() == []

        await session.add_items([{"role": "user", "content": "back"}])
        assert [i.get("content") for i in await session.get_items()] == ["back"]
    finally:
        await session.clear_session()


async def test_sessions_are_isolated_on_real_server(client: aerospike.Client) -> None:
    id_a, id_b = f"it-{uuid.uuid4().hex}", f"it-{uuid.uuid4().hex}"
    session_a = _make_session(client, id_a)
    session_b = _make_session(client, id_b)
    try:
        await session_a.add_items([{"role": "user", "content": "a"}])
        await session_b.add_items([{"role": "user", "content": "b"}])

        assert [i.get("content") for i in await session_a.get_items()] == ["a"]
        assert [i.get("content") for i in await session_b.get_items()] == ["b"]
    finally:
        await session_a.clear_session()
        await session_b.clear_session()


async def test_explicit_ttl_is_set_on_the_real_record(
    client: aerospike.Client, session_id: str
) -> None:
    session = _make_session(client, session_id, ttl=3600)
    try:
        await session.add_items([{"role": "user", "content": "x"}])
        _, meta, _ = client.get((AEROSPIKE_NAMESPACE, _SET_NAME, session_id))
        # Allow scheduling slack; must be a real bounded TTL, not "never expire".
        assert 0 < meta["ttl"] <= 3600
    finally:
        await session.clear_session()


async def test_default_ttl_never_expires_on_the_real_record(
    client: aerospike.Client, session_id: str
) -> None:
    session = _make_session(client, session_id)
    try:
        await session.add_items([{"role": "user", "content": "x"}])
        _, meta, _ = client.get((AEROSPIKE_NAMESPACE, _SET_NAME, session_id))
        assert meta["ttl"] == aerospike.TTL_NEVER_EXPIRE
    finally:
        await session.clear_session()


async def test_ping_against_real_server(client: aerospike.Client, session_id: str) -> None:
    session = _make_session(client, session_id)
    assert await session.ping() is True


async def test_from_config_owns_and_closes_its_own_client(session_id: str) -> None:
    session = AerospikeSession.from_config(
        session_id,
        config={"hosts": [(AEROSPIKE_HOST, AEROSPIKE_PORT)]},
        namespace=AEROSPIKE_NAMESPACE,
        set_name=_SET_NAME,
    )
    try:
        assert await session.ping() is True
        await session.add_items([{"role": "user", "content": "x"}])
    finally:
        await session.clear_session()
        await session.close()

    with pytest.raises(RuntimeError, match="^AerospikeSession is closed$"):
        await session.get_items()
