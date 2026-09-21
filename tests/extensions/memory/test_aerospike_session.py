"""Tests for AerospikeSession using in-process fake client objects.

All tests run without a real Aerospike server — or even the ``aerospike``
package — by injecting a lightweight fake client and fake CDT list-operation
helpers into ``sys.modules`` before the module under test is imported. The
fake reproduces exactly the server behavior observed against a real
Aerospike CE container (see AERO_VALIDATION.md): ``list_append_items``
auto-creates the record; ``list_get_range``, ``list_pop``, and ``list_clear``
raise ``RecordNotFound`` on a missing record; ``list_pop`` on an empty list
raises ``OpNotApplicable``; ``list_get_range`` on an empty list returns
``[]``.

Real-server behavior (Docker CE) is covered separately by
test_aerospike_session_integration.py, which skips when no server is
reachable.
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any
from unittest.mock import patch

import pytest

from agents import Agent, Runner, TResponseInputItem
from agents.memory.session_settings import SessionSettings
from agents.testing import ScriptedModel
from tests.test_responses import get_text_message

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# In-memory fake aerospike client and CDT list-operation helpers
# ---------------------------------------------------------------------------


class FakeRecordNotFound(Exception):
    pass


class FakeOpNotApplicable(Exception):
    pass


_FAKE_TTL_NEVER_EXPIRE = 4294967295


def _fake_list_append_items(bin_name: str, values: list[str]) -> dict[str, Any]:
    return {"op": "list_append_items", "bin": bin_name, "values": values}


def _fake_list_pop(bin_name: str, index: int) -> dict[str, Any]:
    return {"op": "list_pop", "bin": bin_name, "index": index}


def _fake_list_clear(bin_name: str) -> dict[str, Any]:
    return {"op": "list_clear", "bin": bin_name}


def _fake_list_get_range(bin_name: str, index: int, count: int) -> dict[str, Any]:
    return {"op": "list_get_range", "bin": bin_name, "index": index, "count": count}


class FakeAerospikeClient:
    """In-memory substitute for aerospike.Client, one record per key."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config
        self._records: dict[Any, dict[str, list[str]]] = {}
        self._last_write_meta: dict[str, Any] | None = None
        self._closed = False

    def connect(self) -> FakeAerospikeClient:
        return self

    def get(self, key: Any) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        if key not in self._records:
            raise FakeRecordNotFound("record not found")
        return key, {"ttl": _FAKE_TTL_NEVER_EXPIRE, "gen": 1}, dict(self._records[key])

    def operate(
        self,
        key: Any,
        ops: list[dict[str, Any]],
        meta: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        (op,) = ops
        kind = op["op"]
        bin_name = op["bin"]
        result_bins: dict[str, Any] = {}
        self._last_write_meta = meta

        if kind == "list_append_items":
            record = self._records.setdefault(key, {})
            record.setdefault(bin_name, []).extend(op["values"])
            result_bins[bin_name] = len(record[bin_name])
        elif kind == "list_get_range":
            if key not in self._records:
                raise FakeRecordNotFound("record not found")
            values = self._records[key].get(bin_name, [])
            index = op["index"]
            start = index if index >= 0 else max(0, len(values) + index)
            result_bins[bin_name] = values[start : start + op["count"]]
        elif kind == "list_pop":
            if key not in self._records:
                raise FakeRecordNotFound("record not found")
            values = self._records[key].get(bin_name, [])
            if not values:
                raise FakeOpNotApplicable("list is empty")
            result_bins[bin_name] = values.pop(op["index"])
        elif kind == "list_clear":
            if key not in self._records:
                raise FakeRecordNotFound("record not found")
            self._records[key][bin_name] = []
        else:  # pragma: no cover - defensive
            raise AssertionError(f"unhandled fake op: {kind}")

        return key, {"ttl": _FAKE_TTL_NEVER_EXPIRE, "gen": 1}, result_bins

    def remove(self, key: Any) -> None:
        if key not in self._records:
            raise FakeRecordNotFound("record not found")
        del self._records[key]

    def close(self) -> None:
        self._closed = True

    def is_connected(self) -> bool:
        return not self._closed


# ---------------------------------------------------------------------------
# Inject fake aerospike modules before importing the module under test
# ---------------------------------------------------------------------------


_FAKE_MODULE_NAMES = (
    "aerospike",
    "aerospike.exception",
    "aerospike_helpers",
    "aerospike_helpers.operations",
    "aerospike_helpers.operations.list_operations",
)


def _install_fake_aerospike_modules() -> dict[str, types.ModuleType | None]:
    """Inject fakes into sys.modules and return the entries they replaced.

    The real ``aerospike`` package may also be installed (it is required by
    test_aerospike_session_integration.py in the same test run), so the
    original entries — real or absent — are restored immediately after this
    module under test finishes importing, rather than left clobbered for the
    rest of the pytest session.
    """
    originals = {name: sys.modules.get(name) for name in _FAKE_MODULE_NAMES}

    aerospike_mod = types.ModuleType("aerospike")
    aerospike_mod.client = lambda config: FakeAerospikeClient(config)  # type: ignore[attr-defined]
    aerospike_mod.TTL_NEVER_EXPIRE = _FAKE_TTL_NEVER_EXPIRE  # type: ignore[attr-defined]

    exception_mod = types.ModuleType("aerospike.exception")
    exception_mod.RecordNotFound = FakeRecordNotFound  # type: ignore[attr-defined]
    exception_mod.OpNotApplicable = FakeOpNotApplicable  # type: ignore[attr-defined]
    aerospike_mod.exception = exception_mod  # type: ignore[attr-defined]

    helpers_pkg = types.ModuleType("aerospike_helpers")
    operations_pkg = types.ModuleType("aerospike_helpers.operations")
    list_operations_mod = types.ModuleType("aerospike_helpers.operations.list_operations")
    list_operations_mod.list_append_items = _fake_list_append_items  # type: ignore[attr-defined]
    list_operations_mod.list_pop = _fake_list_pop  # type: ignore[attr-defined]
    list_operations_mod.list_clear = _fake_list_clear  # type: ignore[attr-defined]
    list_operations_mod.list_get_range = _fake_list_get_range  # type: ignore[attr-defined]

    sys.modules["aerospike"] = aerospike_mod
    sys.modules["aerospike.exception"] = exception_mod
    sys.modules["aerospike_helpers"] = helpers_pkg
    sys.modules["aerospike_helpers.operations"] = operations_pkg
    sys.modules["aerospike_helpers.operations.list_operations"] = list_operations_mod

    return originals


def _restore_modules(originals: dict[str, types.ModuleType | None]) -> None:
    for name, original in originals.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


def _evict_cached_aerospike_session_module() -> None:
    """Drop any cached agents.extensions.memory.aerospike_session module.

    test_aerospike_session_integration.py imports the same production module
    against the *real* aerospike package. Whichever test file runs first
    leaves its own bound copy cached in sys.modules; without eviction, the
    file that runs second would silently reuse that stale copy (still closed
    over the other file's fake-or-real dependencies) instead of importing
    fresh against the dependencies it is about to install. Only the cache
    entry is evicted here — nothing before this point has run any import of
    the module yet, so there is nothing else to unwind.
    """
    sys.modules.pop("agents.extensions.memory.aerospike_session", None)
    parent_package = sys.modules.get("agents.extensions.memory")
    if parent_package is not None:
        parent_package.__dict__.pop("aerospike_session", None)


_evict_cached_aerospike_session_module()
_original_modules = _install_fake_aerospike_modules()

# Now it's safe to import the module under test. This binds AerospikeSession's
# module to the fakes above. A second test file (the real-server integration
# suite) importing the same production module later during pytest's
# collection phase will overwrite the sys.modules cache entry for it — before
# any test function actually runs — so patch targets below are bound directly
# to this captured module object rather than re-resolved by dotted-string
# path through sys.modules at patch-application time.
import agents.extensions.memory.aerospike_session as aerospike_session_module  # noqa: E402
from agents.extensions.memory.aerospike_session import AerospikeSession  # noqa: E402

# Restore the real aerospike/aerospike_helpers entries (if any) so anything
# else in the process — including a same-session import of the real package —
# is unaffected by this file having stood in fakes for its own import above.
_restore_modules(_original_modules)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_session(session_id: str = "test-session", **kwargs: Any) -> AerospikeSession:
    client = FakeAerospikeClient()
    return AerospikeSession(
        session_id,
        client=client,
        namespace="test",
        **kwargs,
    )


@pytest.fixture
def session() -> AerospikeSession:
    return _make_session()


@pytest.fixture
def agent() -> Agent:
    return Agent(name="test", model=ScriptedModel())


# ---------------------------------------------------------------------------
# Core CRUD tests
# ---------------------------------------------------------------------------


async def test_add_and_get_items(session: AerospikeSession) -> None:
    """Items added to the session are retrievable in insertion order."""
    items: list[TResponseInputItem] = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    await session.add_items(items)

    retrieved = await session.get_items()
    assert len(retrieved) == 2
    assert retrieved[0].get("content") == "Hello"
    assert retrieved[1].get("content") == "Hi there!"


async def test_add_empty_list_is_noop(session: AerospikeSession) -> None:
    """Adding an empty list must not create a record."""
    await session.add_items([])
    assert await session.get_items() == []
    assert session._key not in session._client._records


async def test_get_items_empty_session(session: AerospikeSession) -> None:
    """Retrieving items from a brand-new session returns an empty list."""
    assert await session.get_items() == []


async def test_pop_item_returns_last(session: AerospikeSession) -> None:
    """pop_item must return and remove the most recently added item."""
    items: list[TResponseInputItem] = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]
    await session.add_items(items)

    popped = await session.pop_item()
    assert popped is not None
    assert popped.get("content") == "second"

    remaining = await session.get_items()
    assert len(remaining) == 1
    assert remaining[0].get("content") == "first"


async def test_pop_item_empty_session(session: AerospikeSession) -> None:
    """pop_item on a session with no record must return None."""
    assert await session.pop_item() is None


async def test_pop_item_empty_list_after_clear(session: AerospikeSession) -> None:
    """pop_item on an existing but empty list (post-clear) must return None."""
    await session.add_items([{"role": "user", "content": "x"}])
    await session.clear_session()
    assert await session.pop_item() is None


async def test_clear_session(session: AerospikeSession) -> None:
    """clear_session empties the list bin without erroring on a fresh session."""
    await session.add_items([{"role": "user", "content": "x"}])
    await session.clear_session()

    assert await session.get_items() == []

    await session.add_items([{"role": "user", "content": "after clear"}])
    assert len(await session.get_items()) == 1


async def test_clear_session_on_missing_record_is_noop(session: AerospikeSession) -> None:
    """clear_session on a session with no record must not raise."""
    await session.clear_session()
    assert await session.get_items() == []


async def test_multiple_add_calls_accumulate(session: AerospikeSession) -> None:
    """Multiple add_items calls must append rather than overwrite."""
    await session.add_items([{"role": "user", "content": "1"}])
    await session.add_items([{"role": "user", "content": "2"}])
    await session.add_items([{"role": "user", "content": "3"}])

    items = await session.get_items()
    assert [item.get("content") for item in items] == ["1", "2", "3"]


# ---------------------------------------------------------------------------
# Limit handling
# ---------------------------------------------------------------------------


async def test_get_items_with_explicit_limit(session: AerospikeSession) -> None:
    """An explicit limit returns only the latest N items, in order."""
    for i in range(5):
        await session.add_items([{"role": "user", "content": str(i)}])

    items = await session.get_items(limit=2)
    assert [item.get("content") for item in items] == ["3", "4"]


async def test_get_items_limit_zero(session: AerospikeSession) -> None:
    """A limit of zero must return an empty list without touching the server."""
    await session.add_items([{"role": "user", "content": "x"}])
    assert await session.get_items(limit=0) == []


async def test_get_items_limit_exceeds_count(session: AerospikeSession) -> None:
    """A limit larger than the stored history returns everything available."""
    await session.add_items([{"role": "user", "content": "only"}])
    items = await session.get_items(limit=10)
    assert len(items) == 1


async def test_session_settings_limit_used_as_default() -> None:
    """SessionSettings.limit applies when no explicit limit is passed."""
    session = _make_session(session_settings=SessionSettings(limit=2))
    for i in range(4):
        await session.add_items([{"role": "user", "content": str(i)}])

    items = await session.get_items()
    assert [item.get("content") for item in items] == ["2", "3"]


async def test_explicit_limit_overrides_session_settings() -> None:
    """An explicit get_items(limit=...) overrides SessionSettings.limit."""
    session = _make_session(session_settings=SessionSettings(limit=2))
    for i in range(4):
        await session.add_items([{"role": "user", "content": str(i)}])

    items = await session.get_items(limit=1)
    assert [item.get("content") for item in items] == ["3"]


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


async def test_sessions_are_isolated() -> None:
    """Two sessions on the same client must not see each other's history."""
    client = FakeAerospikeClient()
    s1 = AerospikeSession("a", client=client, namespace="test")
    s2 = AerospikeSession("b", client=client, namespace="test")

    await s1.add_items([{"role": "user", "content": "s1"}])
    await s2.add_items([{"role": "user", "content": "s2"}])

    assert [i.get("content") for i in await s1.get_items()] == ["s1"]
    assert [i.get("content") for i in await s2.get_items()] == ["s2"]


async def test_clear_does_not_affect_other_sessions() -> None:
    """Clearing one session must not remove another session's history."""
    client = FakeAerospikeClient()
    s1 = AerospikeSession("a", client=client, namespace="test")
    s2 = AerospikeSession("b", client=client, namespace="test")

    await s1.add_items([{"role": "user", "content": "keep"}])
    await s2.add_items([{"role": "user", "content": "drop"}])

    await s2.clear_session()

    assert [i.get("content") for i in await s1.get_items()] == ["keep"]
    assert await s2.get_items() == []


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


async def test_unicode_content_roundtrip(session: AerospikeSession) -> None:
    """Non-ASCII content must survive a full add/get roundtrip."""
    await session.add_items([{"role": "user", "content": "こんにちは 🎉"}])
    items = await session.get_items()
    assert items[0].get("content") == "こんにちは 🎉"


async def test_json_special_characters(session: AerospikeSession) -> None:
    """Characters requiring JSON escaping must round-trip correctly."""
    content = 'quote"backslash\\newline\ntab\t'
    await session.add_items([{"role": "user", "content": content}])
    items = await session.get_items()
    assert items[0].get("content") == content


async def test_corrupted_item_is_skipped_in_get_items(session: AerospikeSession) -> None:
    """A malformed JSON entry in the list bin must be skipped, not raised."""
    await session.add_items([{"role": "user", "content": "good"}])
    session._client._records[session._key][session._bin_name].append("{not valid json")
    await session.add_items([{"role": "user", "content": "also good"}])

    items = await session.get_items()
    assert [i.get("content") for i in items] == ["good", "also good"]


async def test_pop_item_skips_corrupt_most_recent(session: AerospikeSession) -> None:
    """pop_item must discard a corrupt tail entry and return the next-most-recent item."""
    await session.add_items([{"role": "user", "content": "real"}])
    session._client._records[session._key][session._bin_name].append("{not valid json")

    popped = await session.pop_item()
    assert popped is not None
    assert popped.get("content") == "real"
    assert await session.get_items() == []


async def test_pop_item_returns_none_when_only_corrupt_items_remain(
    session: AerospikeSession,
) -> None:
    """pop_item must return None, not raise, if every remaining entry is corrupt."""
    session._client._records.setdefault(session._key, {})[session._bin_name] = [
        "{bad",
        "{also bad",
    ]
    assert await session.pop_item() is None


async def test_pop_item_uses_overridden_deserialize_item(session: AerospikeSession) -> None:
    """pop_item must route through the overridable _deserialize_item, like get_items."""
    calls: list[str] = []
    original_deserialize = session._deserialize_item

    async def _tracking_deserialize(raw: str) -> Any:
        calls.append(raw)
        return await original_deserialize(raw)

    session._deserialize_item = _tracking_deserialize  # type: ignore[method-assign]

    await session.add_items([{"role": "user", "content": "hi"}])
    popped = await session.pop_item()

    assert popped is not None and popped.get("content") == "hi"
    assert calls == ['{"role":"user","content":"hi"}']


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------


async def test_default_ttl_never_expires(session: AerospikeSession) -> None:
    """Without an explicit ttl, writes must use TTL_NEVER_EXPIRE."""
    import aerospike

    await session.add_items([{"role": "user", "content": "x"}])
    fake_client: FakeAerospikeClient = session._client
    assert fake_client._last_write_meta == {"ttl": aerospike.TTL_NEVER_EXPIRE}


async def test_explicit_ttl_is_forwarded() -> None:
    """An explicit ttl= must be forwarded on every write."""
    session = _make_session(ttl=3600)
    await session.add_items([{"role": "user", "content": "x"}])
    fake_client: FakeAerospikeClient = session._client
    assert fake_client._last_write_meta == {"ttl": 3600}


async def test_explicit_ttl_is_forwarded_on_pop_item() -> None:
    """pop_item() must forward ttl= like add_items(), not fall back to the default."""
    session = _make_session(ttl=3600)
    await session.add_items([{"role": "user", "content": "x"}])
    await session.pop_item()
    fake_client: FakeAerospikeClient = session._client
    assert fake_client._last_write_meta == {"ttl": 3600}


async def test_explicit_ttl_is_forwarded_on_clear_session() -> None:
    """clear_session() must forward ttl= like add_items(), not fall back to the default."""
    session = _make_session(ttl=3600)
    await session.add_items([{"role": "user", "content": "x"}])
    await session.clear_session()
    fake_client: FakeAerospikeClient = session._client
    assert fake_client._last_write_meta == {"ttl": 3600}


# ---------------------------------------------------------------------------
# Connectivity and lifecycle
# ---------------------------------------------------------------------------


async def test_ping_success(session: AerospikeSession) -> None:
    """ping() must return True for a connected client."""
    assert await session.ping() is True


async def test_ping_failure(session: AerospikeSession) -> None:
    """ping() must return False when checking connectivity raises."""
    session._client.is_connected = lambda: (_ for _ in ()).throw(ConnectionError("down"))
    assert await session.ping() is False


async def test_close_external_client_not_closed() -> None:
    """close() must NOT close a client that was injected externally."""
    client = FakeAerospikeClient()
    s = AerospikeSession("x", client=client, namespace="test")
    assert s._owns_client is False

    await s.close()
    assert not client._closed


async def test_close_owned_client_is_closed() -> None:
    """close() must close a client created by from_config."""
    fake_client = FakeAerospikeClient()
    with patch.object(
        aerospike_session_module.aerospike,  # type: ignore[attr-defined]
        "client",
        return_value=fake_client,
    ):
        s = AerospikeSession.from_config("owned", config={"hosts": []}, namespace="test")
        assert s._owns_client is True

        await s.close()
        assert fake_client._closed


def _make_owned_session(session_id: str = "owned") -> AerospikeSession:
    with patch.object(
        aerospike_session_module.aerospike,  # type: ignore[attr-defined]
        "client",
        return_value=FakeAerospikeClient(),
    ):
        return AerospikeSession.from_config(session_id, config={"hosts": []}, namespace="test")


async def test_closed_operations_raise_runtime_error() -> None:
    """Operations on a closed session must fail instead of running against a released client."""
    session = _make_owned_session()
    await session.add_items([{"role": "user", "content": "hi"}])
    await session.close()

    with pytest.raises(RuntimeError, match="^AerospikeSession is closed$"):
        await session.get_items()
    with pytest.raises(RuntimeError, match="^AerospikeSession is closed$"):
        await session.add_items([{"role": "user", "content": "after close"}])
    with pytest.raises(RuntimeError, match="^AerospikeSession is closed$"):
        await session.pop_item()
    with pytest.raises(RuntimeError, match="^AerospikeSession is closed$"):
        await session.clear_session()
    with pytest.raises(RuntimeError, match="^AerospikeSession is closed$"):
        await session.ping()


async def test_closed_rejects_empty_add_items() -> None:
    """add_items([]) must not bypass the closed check through the empty-list fast path."""
    session = _make_owned_session()
    await session.close()

    with pytest.raises(RuntimeError, match="^AerospikeSession is closed$"):
        await session.add_items([])


async def test_repeated_close_remains_safe() -> None:
    """Repeated close() calls must remain safe for callers."""
    session = _make_owned_session()

    await session.close()
    await session.close()

    with pytest.raises(RuntimeError, match="^AerospikeSession is closed$"):
        await session.get_items()


async def test_external_client_session_stays_usable_after_close() -> None:
    """An injected client is the caller's to manage, so close() must not be terminal."""
    session = _make_session()
    assert session._owns_client is False

    await session.close()

    await session.add_items([{"role": "user", "content": "still works"}])
    assert len(await session.get_items()) == 1


async def test_from_config_forwards_config_and_kwargs() -> None:
    """from_config must build the client from the given config and thread through kwargs."""
    captured: dict[str, Any] = {}

    def _fake_client_factory(config: dict[str, Any]) -> FakeAerospikeClient:
        captured["config"] = config
        return FakeAerospikeClient(config)

    with patch.object(
        aerospike_session_module.aerospike,  # type: ignore[attr-defined]
        "client",
        side_effect=_fake_client_factory,
    ):
        session = AerospikeSession.from_config(
            "owned",
            config={"hosts": [("127.0.0.1", 3000)]},
            namespace="test",
            set_name="custom_set",
        )

    assert captured["config"] == {"hosts": [("127.0.0.1", 3000)]}
    assert session._set_name == "custom_set"


async def test_mutation_cancellation_waits_for_authoritative_outcome() -> None:
    """A cancelled add_items must still let the in-flight write settle before re-raising."""
    session = _make_session()
    real_operate = session._client.operate

    def _slow_operate(*args: Any, **kwargs: Any) -> Any:
        return real_operate(*args, **kwargs)

    with patch.object(session._client, "operate", side_effect=_slow_operate):
        task = asyncio.ensure_future(session.add_items([{"role": "user", "content": "x"}]))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    # The write already reached the fake "server" before cancellation was observed.
    assert len(await session.get_items()) == 1


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------


async def test_runner_integration(agent: Agent) -> None:
    """AerospikeSession must supply conversation history to the Runner."""
    session = _make_session("runner-test")

    assert isinstance(agent.model, ScriptedModel)
    agent.model.enqueue([get_text_message("San Francisco")])
    result1 = await Runner.run(agent, "Where is the Golden Gate Bridge?", session=session)
    assert result1.final_output == "San Francisco"

    agent.model.enqueue([get_text_message("California")])
    result2 = await Runner.run(agent, "What state is it in?", session=session)
    assert result2.final_output == "California"

    last_input = agent.model.calls[-1].input
    assert len(last_input) > 1
    assert any("Golden Gate Bridge" in str(item.get("content", "")) for item in last_input)


async def test_runner_session_isolation(agent: Agent) -> None:
    """Two independent sessions must not bleed history into each other."""
    client = FakeAerospikeClient()
    s1 = AerospikeSession("user-a", client=client, namespace="test")
    s2 = AerospikeSession("user-b", client=client, namespace="test")

    assert isinstance(agent.model, ScriptedModel)
    agent.model.enqueue([get_text_message("I like cats.")])
    await Runner.run(agent, "I like cats.", session=s1)

    agent.model.enqueue([get_text_message("I like dogs.")])
    await Runner.run(agent, "I like dogs.", session=s2)

    agent.model.enqueue([get_text_message("You said you like cats.")])
    result = await Runner.run(agent, "What animal did I mention?", session=s1)
    assert "cats" in result.final_output.lower()
    assert "dogs" not in result.final_output.lower()


async def test_runner_with_session_settings_limit(agent: Agent) -> None:
    """RunConfig.session_settings.limit must cap the history sent to the model."""
    from agents import RunConfig

    session = _make_session("limited")

    assert isinstance(agent.model, ScriptedModel)
    for i in range(4):
        agent.model.enqueue([get_text_message(f"reply {i}")])
        await Runner.run(agent, f"message {i}", session=session)

    agent.model.enqueue([get_text_message("final")])
    await Runner.run(
        agent,
        "final message",
        session=session,
        run_config=RunConfig(session_settings=SessionSettings(limit=2)),
    )
    last_input = agent.model.calls[-1].input
    # 2 history items (limit) + the new user message.
    assert len(last_input) == 3
