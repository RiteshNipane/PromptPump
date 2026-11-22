# Feature flags

Public flag helpers live in `main.py`. `new_nav` defaults to false and reads `APP_FEATURE_NEW_NAV`; `compact_cards` defaults to true and reads `APP_FEATURE_COMPACT_CARDS`. `load_feature_flags` should use `parse_bool`, accept an optional environment mapping, and return only the two documented flag names. Unknown environment variables are ignored.
