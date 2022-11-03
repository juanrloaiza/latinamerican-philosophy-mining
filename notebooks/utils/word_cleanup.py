"""
This modules contains a set of utilities for cleaning
the topics post-hoc. Some of the words were e.g. considered
verbs when they shouldn't.

One example is Poincaré and poincarear.

The full list is being maintained at ./wordlists/word_cleanup.json.
"""
from typing import Dict
from pathlib import Path
import json

ROOT_DIR = Path(__file__).parent.parent.parent.resolve()
WORDLISTS_DIR = ROOT_DIR / "notebooks" / "wordlists"

# I wonder if loading it here is best practice, but we actually want
# to only load it once. Putting it inside the clean_word function
# is a really bad idea.
with open(WORDLISTS_DIR / "word_cleanup.json") as fp:
    POSTHOC_WORD_CLEANER: Dict = json.load(fp)


def clean_word(word: str) -> str:
    """
    Grabs a word and either leaves it be if it isn't
    in word_cleanup.json's keys, or replaces it with
    what word_cleanup.json says it should be replaced.
    """
    return POSTHOC_WORD_CLEANER.get(word, word)


if __name__ == "__main__":
    print(ROOT_DIR)
