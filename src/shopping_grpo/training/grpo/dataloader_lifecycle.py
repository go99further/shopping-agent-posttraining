"""Explicit lifecycle management for stateful dataloader worker iterators."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def shutdown_stateful_dataloaders(
    entries: Iterable[tuple[str, object, object | None]],
) -> tuple[dict[str, Any], ...]:
    """Shut down explicit and cached worker iterators in deterministic order.

    ``StatefulDataLoader`` keeps a persistent iterator on ``_iterator`` while
    non-persistent loaders return an iterator owned by the active ``for`` loop.
    The caller therefore supplies both the explicit iterator (when available)
    and its loader.  Each iterator is shut down at most once, and the loader's
    cached reference is cleared so a repeated call is a true no-op.

    Shutdown exceptions deliberately propagate: lifecycle failures must fail a
    canary instead of being hidden as an ignored ``__del__`` warning.
    """
    receipts: list[dict[str, Any]] = []
    seen_iterators: set[int] = set()
    for name, dataloader, explicit_iterator in entries:
        cached_iterator = getattr(dataloader, "_iterator", None)
        for source, iterator in (
            ("explicit", explicit_iterator),
            ("cached", cached_iterator),
        ):
            if iterator is None:
                continue
            iterator_identity = id(iterator)
            if iterator_identity in seen_iterators:
                continue
            seen_iterators.add(iterator_identity)
            shutdown_workers = getattr(iterator, "_shutdown_workers", None)
            if not callable(shutdown_workers):
                receipts.append(
                    {
                        "loader": str(name),
                        "source": source,
                        "action": "no_worker_shutdown",
                    }
                )
                continue
            if bool(getattr(iterator, "_shutdown", False)):
                action = "already_shutdown"
            else:
                shutdown_workers()
                action = "shutdown"
            if getattr(dataloader, "_iterator", None) is iterator:
                dataloader._iterator = None
            receipts.append(
                {
                    "loader": str(name),
                    "source": source,
                    "action": action,
                }
            )
    return tuple(receipts)
