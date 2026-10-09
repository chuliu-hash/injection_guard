"""风险代码映射测试。"""

from injection_guard.defense.risk_mapping import CATEGORY_NAMES, RISK_LEVEL_MAPPING

HIGH_RISK = ["S2", "S3", "S5", "S9"]
MEDIUM_RISK = ["S4", "S6", "S7", "S8", "S10", "S11", "S12"]


def test_high_risk_codes():
    for code in HIGH_RISK:
        assert RISK_LEVEL_MAPPING[code] == "高风险"


def test_low_risk_code():
    assert RISK_LEVEL_MAPPING["S1"] == "低风险"


def test_medium_risk_codes():
    for code in MEDIUM_RISK:
        assert RISK_LEVEL_MAPPING[code] == "中风险"


def test_category_names_cover_s1_to_s12():
    for i in range(1, 13):
        code = f"S{i}"
        assert code in CATEGORY_NAMES
        assert code in RISK_LEVEL_MAPPING
