"""Regression tests for cycle-dependent view routing.

The property under test is the one the old static partition violated: an
anchor must not be permanently locked to a single view.
"""

from collections import Counter

import pytest

hvd = pytest.importorskip("openpi.training.human_vlm_dataset")

HIGH = hvd.VIEW_HIGH_LEVEL
LOW = hvd.VIEW_LOW_LEVEL


# ---------------------------------------------------------------- schedules

def test_one_to_one_schedule_alternates():
    assert hvd.build_view_schedule({HIGH: 1.0, LOW: 1.0}) == (HIGH, LOW)


def test_three_to_one_schedule_has_span_four():
    sched = hvd.build_view_schedule({HIGH: 3.0, LOW: 1.0})
    assert len(sched) == 4
    assert sched.count(HIGH) == 3
    assert sched.count(LOW) == 1


def test_single_view_schedule_has_span_one():
    assert hvd.build_view_schedule({LOW: 1.0}) == (LOW,)


def test_zero_weight_view_is_dropped():
    assert hvd.build_view_schedule({HIGH: 0.0, LOW: 1.0}) == (LOW,)


def test_schedule_ratio_matches_weights():
    for hi, lo in ((1, 1), (3, 1), (1, 3), (2, 1)):
        sched = hvd.build_view_schedule({HIGH: float(hi), LOW: float(lo)})
        assert sched.count(HIGH) / len(sched) == pytest.approx(hi / (hi + lo))


# ------------------------------------------------------------- stable hash

def test_splitmix64_is_deterministic_and_in_range():
    a = hvd.splitmix64(12345)
    assert a == hvd.splitmix64(12345)
    assert 0 <= a < 2**64


def test_splitmix64_separates_adjacent_inputs():
    # Adjacent anchors must not land in the same phase bucket systematically,
    # or whole contiguous runs of anchors would share a view every cycle.
    lows = sum(hvd.splitmix64(i) % 2 for i in range(4096))
    assert 1800 < lows < 2300, f"splitmix64 parity is skewed: {lows}/4096"


# ---------------------------------------------------- routing on a fixture

class _FakeDataset:
    """Exercises the real routing methods against a controlled anchor layout."""

    def __init__(self, anchors_by_source, view_specs, seed=20260727):
        import math

        self._source_cum_rows = {s: [n] for s, n in anchors_by_source.items()}
        self._view_schedule_by_source = {
            s: hvd.build_view_schedule(w) for s, w in view_specs.items()
        }
        self.num_anchor_rows = sum(anchors_by_source.values())
        self.view_cycle_span = math.lcm(
            *(len(s) for s in self._view_schedule_by_source.values())
        )
        self.num_frames = self.num_anchor_rows * self.view_cycle_span
        self._view_route_seed = seed
        self.view_cycle_mode = "complete_view_cycle"
        self.view_route_phase = 0
        bounds, run = [], 0
        for s in anchors_by_source:
            run += anchors_by_source[s]
            bounds.append((run, s))
        self._bounds = bounds

    def _source_for_anchor(self, anchor_idx):
        for end, source in self._bounds:
            if anchor_idx < end:
                return source
        raise AssertionError(anchor_idx)

    route_view = hvd.HumanVLMPairedOfflineDataset.route_view
    index_space_counts = hvd.HumanVLMPairedOfflineDataset.index_space_counts
    index_space_ratios = hvd.HumanVLMPairedOfflineDataset.index_space_ratios


def _mixture():
    return _FakeDataset(
        {"EgoDex": 300, "EgoLive": 500, "VITRA": 100},
        {
            "EgoDex": {HIGH: 1.0, LOW: 1.0},
            "EgoLive": {HIGH: 1.0, LOW: 1.0},
            "VITRA": {LOW: 1.0},
        },
    )


def test_every_two_level_anchor_reaches_both_views():
    """The exact property the static partition violated."""
    ds = _mixture()
    seen = {}
    for cycle in range(ds.view_cycle_span):
        for slot in range(ds.num_anchor_rows):
            source, view, anchor = ds.route_view(slot, cycle)
            seen.setdefault((source, anchor), set()).add(view)
    for (source, _anchor), views in seen.items():
        if source == "VITRA":
            assert views == {LOW}
        else:
            assert views == {HIGH, LOW}, f"{source} anchor stuck on {views}"


def test_vitra_is_never_high_level():
    ds = _mixture()
    for cycle in range(6):
        for slot in range(ds.num_anchor_rows):
            source, view, _ = ds.route_view(slot, cycle)
            if source == "VITRA":
                assert view == LOW


def test_routing_is_pure_and_resume_stable():
    """Re-deriving a view after a restart must reproduce it exactly."""
    a = _mixture()
    b = _mixture()
    for slot in (0, 7, 299, 300, 799, 899):
        for cycle in (0, 1, 5, 12345):
            assert a.route_view(slot, cycle) == b.route_view(slot, cycle)


def test_analytic_counts_match_brute_force():
    ds = _mixture()
    brute = Counter()
    for cycle in range(ds.view_cycle_span):
        for slot in range(ds.num_anchor_rows):
            source, view, _ = ds.route_view(slot, cycle)
            brute[(source, view)] += 1
    assert dict(brute) == ds.index_space_counts()


def test_global_ratio_is_four_ninths_high_five_ninths_low():
    """3:5:1 sources with 1:1 views on the two-level ones."""
    ds = _FakeDataset(
        {"EgoDex": 300, "EgoLive": 500, "VITRA": 100},
        {
            "EgoDex": {HIGH: 1.0, LOW: 1.0},
            "EgoLive": {HIGH: 1.0, LOW: 1.0},
            "VITRA": {LOW: 1.0},
        },
    )
    r = ds.index_space_ratios()
    assert r["view/high_level"] == pytest.approx(4 / 9)
    assert r["view/low_level"] == pytest.approx(5 / 9)


def test_two_views_do_not_double_a_source_share():
    """The bug this guards: a 2-view source must not get 2x its weight."""
    ds = _FakeDataset(
        {"EgoDex": 300, "EgoLive": 500, "VITRA": 100},
        {
            "EgoDex": {HIGH: 1.0, LOW: 1.0},
            "EgoLive": {HIGH: 1.0, LOW: 1.0},
            "VITRA": {LOW: 1.0},
        },
    )
    r = ds.index_space_ratios()
    assert r["source/EgoDex"] == pytest.approx(300 / 900)
    assert r["source/EgoLive"] == pytest.approx(500 / 900)
    assert r["source/VITRA"] == pytest.approx(100 / 900)


def test_per_source_view_split_is_one_to_one():
    ds = _mixture()
    r = ds.index_space_ratios()
    for src in ("EgoDex", "EgoLive"):
        hi = r[f"source_view/{src}/high_level"]
        lo = r[f"source_view/{src}/low_level"]
        assert hi == pytest.approx(lo)
    assert "source_view/VITRA/high_level" not in r


def test_index_space_is_anchors_times_span():
    ds = _mixture()
    assert ds.view_cycle_span == 2
    assert ds.num_frames == 900 * 2
    assert sum(ds.index_space_counts().values()) == ds.num_frames


def test_seed_changes_assignment_but_not_frequencies():
    a = _mixture()
    b = _FakeDataset(
        {"EgoDex": 300, "EgoLive": 500, "VITRA": 100},
        {
            "EgoDex": {HIGH: 1.0, LOW: 1.0},
            "EgoLive": {HIGH: 1.0, LOW: 1.0},
            "VITRA": {LOW: 1.0},
        },
        seed=999,
    )
    differing = sum(
        a.route_view(s, 0)[1] != b.route_view(s, 0)[1] for s in range(900)
    )
    assert differing > 0, "seed had no effect"
    assert a.index_space_counts() == b.index_space_counts()


def test_three_to_one_still_covers_both_views_within_span():
    ds = _FakeDataset(
        {"EgoDex": 200}, {"EgoDex": {HIGH: 3.0, LOW: 1.0}}
    )
    assert ds.view_cycle_span == 4
    seen = {}
    for cycle in range(ds.view_cycle_span):
        for slot in range(ds.num_anchor_rows):
            _s, view, anchor = ds.route_view(slot, cycle)
            seen.setdefault(anchor, Counter())[view] += 1
    for anchor, counts in seen.items():
        assert counts[HIGH] == 3, f"anchor {anchor}: {counts}"
        assert counts[LOW] == 1, f"anchor {anchor}: {counts}"
