# Text helpers

Public text helpers live in `main.py`. Normalization collapses whitespace and uses Unicode-aware case folding. Slugs keep letters and numbers, convert groups of other characters to a single hyphen, and return an empty string when the input has no letters or numbers.
