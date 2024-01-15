"""This script reads correction counts and produces a csv file to analyze."""

import json
from utils.download.parsertools.dictionary import PhilosophyDictionary, RAEDictionary

dictionary = RAEDictionary()
phil = PhilosophyDictionary()

with open("notebooks/utils/dictionaries/correction_counts.json") as fp:
    counts = json.load(fp)

with open("notebooks/utils/dictionaries/corrections.json") as fp:
    corrections = json.load(fp)

data = []
for word, correction in corrections.items():
    if (
        1 < dictionary.spell.word_frequency[correction] < 100
        and phil.spell.word_frequency[correction] == 0
        and counts[word] > 20
    ):
        data.append(
            {
                "word": word,
                "correction": correction,
                "times corrected": counts[word],
                "frequency in RAE dict": dictionary.spell.word_frequency[correction],
            }
        )

import pandas as pd

df = pd.DataFrame(data)
print(df.sort_values("times corrected", ascending=False))
df.to_csv("most_corrected.csv")
