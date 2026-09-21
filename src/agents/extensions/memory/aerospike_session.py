"""Aerospike-powered Session backend.

Requires the ``aerospike`` client package. Install it with::

    pip install openai-agents[aerospike]

Usage::

    from agents.extensions.memory import AerospikeSession

    # Create from a client config dict
    session = AerospikeSession.from_config(
        session_id="user-123",
        config={"hosts": [("127.0.0.1", 3000)]},
        namespace="test",
    )

    # Or pass an existing, already-connected client that your application
    # already manages
    import aerospike

    client = aerospike.client({"hosts": [("127.0.0.1", 3000)]}).connect()
    session = AerospikeSession(
        session_id="user-123",
        client=client,
        namespace="test",
    )

    await Runner.run(agent, "Hello", session=session)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ._optional_imports import raise_optional_dependency_error

try:
    import aerospike
    from aerospike import exception as aerospike_exception
    from aerospike_helpers.operations import list_operations
except ImportError as e:
    raise_optional_dependency_error(
        "AerospikeSession",
        dependency_name="aerospike",
        extra_name="aerospike",
        cause=e,
    )

from ...items import TResponseInputItem
from ...memory.session import SessionABC
from ...memory.session_settings import (
    SessionSettings,
    coerce_session_settings,
    resolve_session_limit,
)
from ...memory.sqlite_session import _await_mutation


class AerospikeSession(SessionABC):
    """Aerospike implementation of [`Session`][agents.memory.session.Session].

    Conversation history for a session is stored in a single Aerospike record,
    keyed by ``session_id``, in a List CDT bin. Every read and write is a
    single atomic ``operate()`` call on that one record, so no client-side
    locking or generation bookkeeping is required: Aerospike serializes
    concurrent operations on the same key at the server.

    The underlying ``aerospike`` client is a synchronous C extension with no
    native asyncio support, so every call is dispatched through
    ``asyncio.to_thread``. Mutating calls are additionally wrapped in
    ``_await_mutation`` (the same internal helper used by
    [`SQLiteSession`][agents.memory.sqlite_session.SQLiteSession] and
    [`MongoDBSession`][agents.extensions.memory.mongodb_session.MongoDBSession])
    so that a cancelled caller does not lose visibility into a write that
    already reached the server.
    """

    session_settings: SessionSettings | None = None

    def __init__(
        self,
        session_id: str,
        *,
        client: Any,
        namespace: str,
        set_name: str = "agent_sessions",
        bin_name: str = "items",
        ttl: int | None = None,
        session_settings: SessionSettings | dict[str, Any] | None = None,
    ):
        """Initialize a new AerospikeSession.

        Args:
            session_id: Unique identifier for the conversation.
            client: A pre-configured, already-connected ``aerospike.Client``.
            namespace: Aerospike namespace to store session records in.
            set_name: Aerospike set name for session records. Defaults to
                ``"agent_sessions"``.
            bin_name: Name of the List CDT bin that holds the serialized
                conversation items. Defaults to ``"items"``. Aerospike bin
                names are capped at 14 characters.
            ttl: Time-to-live in seconds for the session record. If ``None``,
                the record never expires. Requires the namespace to have
                ``nsup-period`` enabled to take effect.
            session_settings: Optional session configuration. When ``None`` a
                default [`SessionSettings`][agents.memory.session_settings.SessionSettings]
                is used (no item limit).
        """
        self.session_id = session_id
        self.session_settings = (
            coerce_session_settings(session_settings)
            if session_settings is not None
            else SessionSettings()
        )
        self._client = client
        self._namespace = namespace
        self._set_name = set_name
        self._bin_name = bin_name
        self._ttl = ttl
        self._owns_client = False
        self._closed = False

        self._key = (namespace, set_name, session_id)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        session_id: str,
        *,
        config: dict[str, Any],
        namespace: str,
        session_settings: SessionSettings | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AerospikeSession:
        """Create a session from an Aerospike client config dict.

        Args:
            session_id: Conversation ID.
            config: Client config forwarded to ``aerospike.client()``, e.g.
                ``{"hosts": [("127.0.0.1", 3000)]}``.
            namespace: Aerospike namespace to store session records in.
            session_settings: Optional session configuration settings.
            **kwargs: Additional keyword arguments forwarded to the main
                constructor (e.g. ``set_name``, ``bin_name``, ``ttl``).

        Returns:
            An [`AerospikeSession`][agents.extensions.memory.aerospike_session.AerospikeSession]
                connected to the specified Aerospike cluster.
        """
        client = aerospike.client(config).connect()
        session = cls(
            session_id,
            client=client,
            namespace=namespace,
            session_settings=session_settings,
            **kwargs,
        )
        session._owns_client = True
        return session

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    async def _serialize_item(self, item: TResponseInputItem) -> str:
        """Serialize an item to a JSON string. Can be overridden by subclasses."""
        return json.dumps(item, separators=(",", ":"))

    async def _deserialize_item(self, raw: str) -> TResponseInputItem:
        """Deserialize a JSON string to an item. Can be overridden by subclasses."""
        return json.loads(raw)  # type: ignore[no-any-return]

    def _check_not_closed(self) -> None:
        """Raise if the session has already been closed."""
        if self._closed:
            raise RuntimeError("AerospikeSession is closed")

    def _write_meta(self) -> dict[str, Any]:
        """Per-call write metadata, mapping ``ttl=None`` to "never expires"."""
        return {"ttl": aerospike.TTL_NEVER_EXPIRE if self._ttl is None else self._ttl}

    # ------------------------------------------------------------------
    # Session protocol implementation
    # ------------------------------------------------------------------

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        """Retrieve the conversation history for this session.

        Args:
            limit: Maximum number of items to retrieve. When ``None``, the
                effective limit is taken from :attr:`session_settings`.
                If that is also ``None``, all items are returned.
                The returned list is always in chronological (oldest-first)
                order.

        Returns:
            List of input items representing the conversation history.
        """
        self._check_not_closed()
        session_limit = resolve_session_limit(limit, self.session_settings)

        if session_limit is not None and session_limit <= 0:
            return []

        def _get_items_sync() -> list[str]:
            try:
                if session_limit is None:
                    _, _, bins = self._client.get(self._key)
                else:
                    ops = [
                        list_operations.list_get_range(
                            self._bin_name, -session_limit, session_limit
                        )
                    ]
                    _, _, bins = self._client.operate(self._key, ops)
            except aerospike_exception.RecordNotFound:
                return []
            raw_items = bins.get(self._bin_name, [])
            return raw_items if isinstance(raw_items, list) else []

        raw_items = await asyncio.to_thread(_get_items_sync)

        items: list[TResponseInputItem] = []
        for raw in raw_items:
            try:
                items.append(await self._deserialize_item(raw))
            except (json.JSONDecodeError, TypeError):
                # Skip corrupted or malformed entries.
                continue
        return items

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        """Add new items and wait until the batch outcome is known."""
        self._check_not_closed()
        if not items:
            return
        serialized_items = [await self._serialize_item(item) for item in items]

        def _add_items_sync() -> None:
            ops = [list_operations.list_append_items(self._bin_name, serialized_items)]
            self._client.operate(self._key, ops, self._write_meta())

        await _await_mutation(asyncio.to_thread(_add_items_sync))

    async def pop_item(self) -> TResponseInputItem | None:
        """Remove the most recent item after the destructive claim settles.

        Returns:
            The most recent item if it exists, ``None`` if the session is
            empty. Corrupted entries are silently discarded and the
            next-most-recent item is popped instead, so one bad record
            cannot make a non-empty session look empty.
        """
        self._check_not_closed()

        def _pop_raw_sync() -> str:
            ops = [list_operations.list_pop(self._bin_name, -1)]
            _, _, bins = self._client.operate(self._key, ops, self._write_meta())
            return bins.get(self._bin_name)  # type: ignore[no-any-return]

        while True:
            try:
                raw = await _await_mutation(asyncio.to_thread(_pop_raw_sync))
            except (aerospike_exception.RecordNotFound, aerospike_exception.OpNotApplicable):
                return None
            try:
                return await self._deserialize_item(raw)
            except (json.JSONDecodeError, TypeError):
                # Corrupt — drop it and try the next-most-recent item.
                continue

    async def clear_session(self) -> None:
        """Clear history after the authoritative delete settles."""
        self._check_not_closed()

        def _clear_session_sync() -> None:
            try:
                ops = [list_operations.list_clear(self._bin_name)]
                self._client.operate(self._key, ops, self._write_meta())
            except aerospike_exception.RecordNotFound:
                # Already empty.
                pass

        await _await_mutation(asyncio.to_thread(_clear_session_sync))

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying Aerospike connection.

        Only closes the client if this session owns it (i.e. it was created
        via :meth:`from_config`). If the client was injected externally the
        caller is responsible for managing its lifecycle and this is a no-op.
        """
        if not self._owns_client:
            return
        self._closed = True
        await asyncio.to_thread(self._client.close)

    async def ping(self) -> bool:
        """Test Aerospike connectivity.

        Returns:
            ``True`` if the client reports an active cluster connection,
            ``False`` otherwise.
        """
        self._check_not_closed()
        try:
            return bool(await asyncio.to_thread(self._client.is_connected))
        except Exception:
            return False
