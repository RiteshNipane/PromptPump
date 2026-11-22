from main import load_feature_flags


def test_defaults_match_documentation():
    assert load_feature_flags({}) == {"new_nav": False, "compact_cards": True}


def test_documented_environment_variables_use_existing_parser():
    env = {"APP_FEATURE_NEW_NAV": " YES ", "APP_FEATURE_COMPACT_CARDS": "off"}
    assert load_feature_flags(env) == {"new_nav": True, "compact_cards": False}


def test_invalid_values_fall_back_to_each_flag_default():
    env = {"APP_FEATURE_NEW_NAV": "maybe", "APP_FEATURE_COMPACT_CARDS": "unknown"}
    assert load_feature_flags(env) == {"new_nav": False, "compact_cards": True}


def test_only_documented_flags_are_returned():
    flags = load_feature_flags({"APP_FEATURE_NEW_NAV": "true", "APP_FEATURE_OTHER": "true"})
    assert flags == {"new_nav": True, "compact_cards": True}
