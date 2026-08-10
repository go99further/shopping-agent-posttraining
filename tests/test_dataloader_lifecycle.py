from __future__ import annotations

import unittest

from torch.utils.data import DataLoader

from shopping_grpo.training.grpo.dataloader_lifecycle import (
    shutdown_stateful_dataloaders,
)


class _FakeIterator:
    def __init__(self):
        self._shutdown = False
        self.shutdown_calls = 0

    def _shutdown_workers(self):
        self.shutdown_calls += 1
        self._shutdown = True


class _FakeLoader:
    def __init__(self, iterator=None):
        self._iterator = iterator


class DataloaderLifecycleTest(unittest.TestCase):
    def test_fake_explicit_and_cached_iterator_shutdown_is_ordered_and_idempotent(self):
        train_iterator = _FakeIterator()
        validation_iterator = _FakeIterator()
        train_loader = _FakeLoader(train_iterator)
        validation_loader = _FakeLoader(validation_iterator)

        receipts = shutdown_stateful_dataloaders(
            (
                ("train", train_loader, train_iterator),
                ("validation", validation_loader, None),
            )
        )
        self.assertEqual(
            receipts,
            (
                {"loader": "train", "source": "explicit", "action": "shutdown"},
                {"loader": "validation", "source": "cached", "action": "shutdown"},
            ),
        )
        self.assertEqual(train_iterator.shutdown_calls, 1)
        self.assertEqual(validation_iterator.shutdown_calls, 1)
        self.assertIsNone(train_loader._iterator)
        self.assertIsNone(validation_loader._iterator)
        self.assertEqual(
            shutdown_stateful_dataloaders(
                (
                    ("train", train_loader, train_iterator),
                    ("validation", validation_loader, None),
                )
            ),
            (
                {
                    "loader": "train",
                    "source": "explicit",
                    "action": "already_shutdown",
                },
            ),
        )
        self.assertEqual(train_iterator.shutdown_calls, 1)

    def test_real_multiprocessing_iterator_shutdown_is_idempotent(self):
        loader = DataLoader(range(8), batch_size=2, num_workers=1, persistent_workers=True)
        iterator = iter(loader)
        self.assertEqual(next(iterator).tolist(), [0, 1])
        first = shutdown_stateful_dataloaders((("train", loader, iterator),))
        second = shutdown_stateful_dataloaders((("train", loader, iterator),))
        self.assertEqual(first[0]["action"], "shutdown")
        self.assertEqual(second[0]["action"], "already_shutdown")
        self.assertIsNone(loader._iterator)

    def test_shutdown_exception_is_not_suppressed(self):
        class BrokenIterator:
            _shutdown = False

            def _shutdown_workers(self):
                raise RuntimeError("worker shutdown failed")

        with self.assertRaisesRegex(RuntimeError, "worker shutdown failed"):
            shutdown_stateful_dataloaders(
                (("train", _FakeLoader(), BrokenIterator()),)
            )


if __name__ == "__main__":
    unittest.main()
