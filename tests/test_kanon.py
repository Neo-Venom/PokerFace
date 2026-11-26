"""k-anonymity / l-diversity planner over distinct patient clusters (synthetic clusters only)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import Policy  # noqa: E402
from src.privacy.kanon import (  # noqa: E402
    _ClusterQI,
    age_band_ladder,
    band_bounds,
    condition_keys,
    plan_quasi_identifiers,
)


def _q(key: str, age: int, gender: str = "F", district: str = "dist", pin3: str = "", conds: tuple[str, ...] = ()) -> _ClusterQI:
    return _ClusterQI(key, gender, (age, age), district, pin3, frozenset(conds))


class TestLadder(unittest.TestCase):
    def test_every_band_is_truthful(self) -> None:
        for lo, hi in ((0, 0), (5, 5), (17, 19), (33, 41), (59, 61), (84, 90), (101, 115)):
            for band in age_band_ladder(lo, hi):
                b_lo, b_hi = band_bounds(band)
                self.assertTrue(b_lo <= lo and hi <= b_hi, (lo, hi, band))

    def test_minors_stay_minor_and_adult_band_is_closed(self) -> None:
        self.assertEqual(age_band_ladder(7, 7)[-1], "0-17")
        top = age_band_ladder(19, 19)[-1]
        self.assertEqual(top, "18-120")  # never "18+": parsers read "Age: 18+" as exactly 18
        self.assertIn("60+", age_band_ladder(70, 70))

    def test_conditions_ignore_complaint_durations(self) -> None:
        keys = condition_keys({"E11.9", "Type 2 diabetes mellitus", "c/o fever x 5 days", "7 din se bukhar"})
        self.assertEqual(keys, {"E11.9", "type 2 diabetes mellitus"})


class TestPlanner(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = Policy()

    def test_counts_patients_not_rows_and_keeps_dense_classes_narrow(self) -> None:
        qis = [_q(f"c{i}", 41 + (i % 3)) for i in range(6)]
        bands, _pin, levels = plan_quasi_identifiers(qis, self.policy)
        self.assertTrue(all(levels[q.key] == 0 for q in qis), bands)
        self.assertEqual(bands["c0"], "40-44")

    def test_outlier_of_any_age_is_widened_truthfully(self) -> None:
        qis = [_q(f"c{i}", 40 + i) for i in range(6)] + [_q("odd", 23)]
        bands, _pin, _lv = plan_quasi_identifiers(qis, self.policy)
        lo, hi = band_bounds(bands["odd"])
        self.assertTrue(lo <= 23 <= hi)
        self.assertGreater(hi - lo, 20)

    def test_rare_condition_needs_a_larger_class(self) -> None:
        common = [_q(f"c{i}", 40 + i, conds=("I10",)) for i in range(7)]
        rare = _q("rare", 42, conds=("takayasu arteritis",))
        bands, _pin, levels = plan_quasi_identifiers(common + [rare], self.policy)
        self.assertGreater(levels["rare"], levels["c2"])

    def test_pin_prefix_released_only_when_class_stays_k_anonymous(self) -> None:
        qis = [_q(f"a{i}", 40 + i, pin3="226") for i in range(6)] + [_q("b", 42, pin3="227")]
        _bands, keep_pin, _lv = plan_quasi_identifiers(qis, self.policy)
        self.assertTrue(keep_pin["a0"])
        self.assertFalse(keep_pin["b"])

    def test_deterministic(self) -> None:
        qis = [_q(f"c{i}", (i * 7) % 90, gender="MF"[i % 2], district=f"d{i % 3}") for i in range(40)]
        self.assertEqual(plan_quasi_identifiers(qis, self.policy), plan_quasi_identifiers(list(reversed(qis)), self.policy))


if __name__ == "__main__":
    unittest.main()
