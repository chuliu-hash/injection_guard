"""confidence_weighted_vote 纯函数测试。"""

from injection_guard.defense.confidence_vote import confidence_weighted_vote


def test_empty_returns_neutral():
    p, c, diag = confidence_weighted_vote([])
    assert p == 0.5
    assert c == 0.5
    assert diag["U_inter"] == 0.0


def test_all_high_attack():
    p, _c, _ = confidence_weighted_vote([0.95, 0.9, 0.92])
    assert p > 0.85


def test_all_low_attack():
    p, _c, _ = confidence_weighted_vote([0.05, 0.1, 0.08])
    assert p < 0.15


def test_all_neutral_stays_near_half():
    p, _c, _ = confidence_weighted_vote([0.5, 0.5, 0.5])
    assert abs(p - 0.5) < 1e-6


def test_directional_disagreement_dampens_toward_half():
    p_agree, _, _ = confidence_weighted_vote([0.9, 0.9])
    p_disagree, _, diag = confidence_weighted_vote([0.9, 0.1])

    assert p_agree > p_disagree
    assert 0.4 < diag["ratio_pos"] < 0.6
    assert abs(p_disagree - 0.5) < 0.05


def test_confidence_score_in_unit_range():
    for probs in ([0.9, 0.8], [0.5, 0.5], [0.1, 0.2, 0.9]):
        _p, c, _ = confidence_weighted_vote(probs)
        assert 0.0 <= c <= 1.0
