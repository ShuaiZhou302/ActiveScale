"""Canonical action-layout contract tests.

Packs store [head, left, right]; the model consumes pi0.5's canonical
[left, right, head]. Sentinel values make an accidental block swap visible.
"""

import numpy as np
import pytest

from openpi.training.human_vlm_dataset import (
    ACTION_DIM_21,
    MODEL_ACTION_DIM,
    MODEL_ACTION_LAYOUT,
    PACKED_ACTION_LAYOUT,
    POSE7,
    permute_packed_to_model,
    validate_action_layouts,
)

HEAD, LEFT, RIGHT = 100.0, 200.0, 300.0


def _sentinel_row(dim=ACTION_DIM_21):
    """[head..., left..., right...] with a distinct decade per block."""
    row = np.zeros(dim, dtype=np.float32)
    for i in range(POSE7):
        row[i] = HEAD + i
        row[POSE7 + i] = LEFT + i
        row[2 * POSE7 + i] = RIGHT + i
    return row


def test_permutation_moves_each_block_to_its_canonical_slot():
    out = permute_packed_to_model(_sentinel_row())
    assert np.allclose(out[0:POSE7], [LEFT + i for i in range(POSE7)])
    assert np.allclose(out[POSE7 : 2 * POSE7], [RIGHT + i for i in range(POSE7)])
    assert np.allclose(out[2 * POSE7 : 3 * POSE7], [HEAD + i for i in range(POSE7)])


def test_permutation_preserves_within_block_order():
    """Blocks move as units; the 7 components inside must not be shuffled."""
    out = permute_packed_to_model(_sentinel_row())
    for block_start in (0, POSE7, 2 * POSE7):
        block = out[block_start : block_start + POSE7]
        assert np.allclose(np.diff(block), 1.0)


def test_permutation_preserves_padding_beyond_21():
    row = np.zeros(MODEL_ACTION_DIM, dtype=np.float32)
    row[:ACTION_DIM_21] = _sentinel_row()
    row[ACTION_DIM_21:] = -7.0
    out = permute_packed_to_model(row)
    assert np.allclose(out[ACTION_DIM_21:], -7.0)


def test_state_action_and_mask_are_permuted_identically():
    """A mask that disagrees with its action would silently mis-supervise."""
    state = _sentinel_row()
    action = np.stack([_sentinel_row() + t for t in range(5)])
    mask = np.zeros((5, ACTION_DIM_21), dtype=bool)
    mask[:, POSE7 : 2 * POSE7] = True  # only the left block is valid

    p_state = permute_packed_to_model(state)
    p_action = permute_packed_to_model(action)
    p_mask = permute_packed_to_model(mask)

    # Left block landed at slot 0 in all three.
    assert np.allclose(p_state[0:POSE7], [LEFT + i for i in range(POSE7)])
    assert np.allclose(p_action[0, 0:POSE7], [LEFT + i for i in range(POSE7)])
    assert p_mask[:, 0:POSE7].all()
    assert not p_mask[:, POSE7:].any()


def test_permutation_is_a_pure_reordering():
    row = _sentinel_row()
    out = permute_packed_to_model(row)
    assert sorted(out.tolist()) == sorted(row.tolist())


def test_permutation_does_not_mutate_input():
    row = _sentinel_row()
    before = row.copy()
    permute_packed_to_model(row)
    assert np.array_equal(row, before)


def test_permutation_rejects_too_few_dims():
    with pytest.raises(ValueError, match="at least 21"):
        permute_packed_to_model(np.zeros(20, dtype=np.float32))


def test_unknown_layout_fails_closed():
    validate_action_layouts(PACKED_ACTION_LAYOUT, MODEL_ACTION_LAYOUT)
    with pytest.raises(ValueError, match="unsupported action layout"):
        validate_action_layouts("left_right_head_pose7", MODEL_ACTION_LAYOUT)
    with pytest.raises(ValueError, match="unsupported action layout"):
        validate_action_layouts(PACKED_ACTION_LAYOUT, "head_left_right_pose7")
    with pytest.raises(ValueError, match="unsupported action layout"):
        validate_action_layouts("something_else", "another_thing")


# --- quaternion contract ------------------------------------------------------

import numpy as _np  # noqa: E402

from openpi.training.human_vlm_dataset import (  # noqa: E402
    QUAT_IN_POSE7,
    unit_normalize_quaternions,
    unwrap_action_quaternion_signs,
)

_Q = QUAT_IN_POSE7


def _pose_row(quats):
    """Build a canonical 21D row from three xyzw quaternions."""
    row = np.zeros(ACTION_DIM_21, dtype=np.float32)
    for block, q in enumerate(quats):
        s = block * POSE7
        row[s + _Q.start : s + _Q.stop] = q
    return row


def test_unit_normalize_scales_each_block_quaternion():
    row = _pose_row([[0, 0, 0, 2.0], [0, 4.0, 0, 0], [0, 0, 3.0, 0]])
    out = unit_normalize_quaternions(row)
    for block in range(3):
        s = block * POSE7
        assert abs(np.linalg.norm(out[s + _Q.start : s + _Q.stop]) - 1.0) < 1e-6


def test_unwrapping_flips_frames_that_oppose_the_previous_one():
    state = _pose_row([[0, 0, 0, 1.0]] * 3)
    # Frame 1 is the same rotation written with the opposite sign.
    actions = np.stack([_pose_row([[0, 0, 0, 1.0]] * 3), _pose_row([[0, 0, 0, -1.0]] * 3)])
    out = unwrap_action_quaternion_signs(actions, state)
    for block in range(3):
        s = block * POSE7
        assert out[1, s + _Q.stop - 1] > 0, "opposite-sign frame should be flipped back"
        assert float(np.dot(out[0, s + _Q.start : s + _Q.stop], out[1, s + _Q.start : s + _Q.stop])) > 0


def test_unwrapping_is_anchored_on_the_current_state():
    """Frame 0 aligns to the state quaternion, not to an arbitrary convention."""
    state = _pose_row([[0, 0, 0, -1.0]] * 3)  # state carries a negative-w sign
    actions = _pose_row([[0, 0, 0, 1.0]] * 3)[None]
    out = unwrap_action_quaternion_signs(actions, state)
    for block in range(3):
        s = block * POSE7
        assert out[0, s + _Q.stop - 1] < 0, "frame 0 should follow the state's sign"


def test_unwrapping_allows_negative_qw():
    """Temporal continuity and a global qw>=0 convention cannot both hold."""
    state = _pose_row([[0, 0, 0, 1.0]] * 3)
    # A rotation sweeping through qw = 0 continues into negative w.
    seq = []
    for w in (0.9, 0.5, 0.0, -0.5):
        x = float(np.sqrt(max(0.0, 1.0 - w * w)))
        seq.append(_pose_row([[x, 0, 0, w]] * 3))
    out = unwrap_action_quaternion_signs(np.stack(seq), state)
    assert (out[:, _Q.stop - 1] < 0).any(), "unwrapped trajectory should be allowed to pass qw<0"
    for block in range(3):
        s = block * POSE7
        q = out[:, s + _Q.start : s + _Q.stop]
        dots = np.sum(q[1:] * q[:-1], axis=-1)
        assert (dots >= 0).all(), "no sign flip should remain after unwrapping"


def test_unwrapping_preserves_rotation_and_translation():
    rng = np.random.default_rng(0)
    state = _pose_row([[0, 0, 0, 1.0]] * 3)
    raw = rng.normal(size=(6, ACTION_DIM_21)).astype(np.float32)
    out = unwrap_action_quaternion_signs(raw, state)
    for block in range(3):
        s = block * POSE7
        # Translation is untouched.
        assert np.allclose(out[:, s : s + 3], raw[:, s : s + 3])
        # Each quaternion still denotes the same rotation (up to sign).
        raw_q = raw[:, s + _Q.start : s + _Q.stop]
        raw_q = raw_q / np.linalg.norm(raw_q, axis=-1, keepdims=True)
        out_q = out[:, s + _Q.start : s + _Q.stop]
        assert np.allclose(np.abs(np.sum(raw_q * out_q, axis=-1)), 1.0, atol=1e-5)


# --- mask-aware unwrapping ----------------------------------------------------


def _mask_for(valid_flags):
    """[T, 21] mask that marks whole timesteps valid/invalid."""
    m = np.zeros((len(valid_flags), ACTION_DIM_21), dtype=bool)
    for t, v in enumerate(valid_flags):
        m[t, :] = v
    return m


def _quat_seq(ws):
    """Frames whose quaternion is [x, 0, 0, w] with x chosen for unit norm."""
    rows = []
    for w in ws:
        x = float(np.sqrt(max(0.0, 1.0 - w * w)))
        rows.append(_pose_row([[x, 0.0, 0.0, w]] * 3))
    return np.stack(rows)


def test_invalid_frame_does_not_influence_a_later_frames_sign():
    """valid = [T, T, F, T, T]: the masked frame must not act as a reference."""
    valid = [True, True, False, True, True]
    state = _pose_row([[0.0, 0.0, 0.0, 1.0]] * 3)
    seq = _quat_seq([0.9, 0.8, 0.7, 0.6, 0.5])
    # Make the invalid frame point the opposite way; a mask-blind unwrapper
    # would adopt it as the reference and flip frame 3.
    seq[2] = -seq[2]

    out = unwrap_action_quaternion_signs(seq.copy(), state, _mask_for(valid))
    masked_out = unwrap_action_quaternion_signs(seq.copy(), state, None)

    for block in range(3):
        s = block * POSE7
        q = out[:, s + _Q.start : s + _Q.stop]
        # Frame 3 follows frame 1 (the previous VALID frame), staying positive.
        assert float(np.dot(q[3], q[1])) > 0
    # Without the mask the invalid frame drags the sign, which is the bug.
    assert not np.allclose(out, masked_out)


def test_post_unwrapping_flip_rate_over_valid_frames_is_exactly_zero():
    valid = [True, True, False, True, True]
    state = _pose_row([[0.0, 0.0, 0.0, 1.0]] * 3)
    seq = _quat_seq([0.9, 0.4, 0.2, -0.3, -0.8])
    seq[2] = -seq[2]
    out = unwrap_action_quaternion_signs(seq, state, _mask_for(valid))

    idx = [t for t, v in enumerate(valid) if v]
    for block in range(3):
        s = block * POSE7
        q = out[idx][:, s + _Q.start : s + _Q.stop]
        dots = np.sum(q[1:] * q[:-1], axis=-1)
        assert (dots >= 0).all(), f"block {block} flip rate over valid frames is not zero"


def test_invalid_frames_are_left_untouched():
    valid = [True, False, True]
    state = _pose_row([[0.0, 0.0, 0.0, 1.0]] * 3)
    seq = _quat_seq([0.9, -0.9, 0.8])
    before = seq.copy()
    out = unwrap_action_quaternion_signs(seq, state, _mask_for(valid))
    for block in range(3):
        s = block * POSE7
        # The masked frame keeps its written sign; it is not a training target.
        assert np.allclose(
            out[1, s + _Q.start : s + _Q.stop],
            before[1, s + _Q.start : s + _Q.stop] / np.linalg.norm(before[1, s + _Q.start : s + _Q.stop]),
        )


def test_unwrapping_is_per_block_independent():
    """A masked left wrist must not affect the head block's chain."""
    state = _pose_row([[0.0, 0.0, 0.0, 1.0]] * 3)
    seq = _quat_seq([0.9, 0.8, 0.7])
    mask = np.ones((3, ACTION_DIM_21), dtype=bool)
    mask[1, 0:POSE7] = False  # only the left block is invalid at t=1
    out = unwrap_action_quaternion_signs(seq.copy(), state, mask)
    full = unwrap_action_quaternion_signs(seq.copy(), state, np.ones_like(mask))
    # Head and right blocks are unaffected by the left block's mask.
    for s in (POSE7, 2 * POSE7):
        assert np.allclose(out[:, s : s + POSE7], full[:, s : s + POSE7])
