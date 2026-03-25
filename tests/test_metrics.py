from __future__ import annotations

import os
import unittest
from unittest import mock

from prismatic.training.metrics import WeightsBiasesTracker


class WeightsBiasesTrackerTest(unittest.TestCase):
    def test_finalize_does_not_sleep_by_default(self) -> None:
        with mock.patch("prismatic.training.metrics.overwatch.is_rank_zero", return_value=True):
            with mock.patch("prismatic.training.metrics.wandb.finish") as finish:
                with mock.patch("prismatic.training.metrics.time.sleep") as sleep:
                    WeightsBiasesTracker.finalize()

        finish.assert_called_once()
        sleep.assert_not_called()

    def test_finalize_uses_opt_in_wait(self) -> None:
        with mock.patch.dict(os.environ, {"PRISMATIC_WANDB_FINALIZE_WAIT_SECS": "1.5"}, clear=False):
            with mock.patch("prismatic.training.metrics.overwatch.is_rank_zero", return_value=True):
                with mock.patch("prismatic.training.metrics.wandb.finish"):
                    with mock.patch("prismatic.training.metrics.time.sleep") as sleep:
                        WeightsBiasesTracker.finalize()

        sleep.assert_called_once_with(1.5)


if __name__ == "__main__":
    unittest.main()
