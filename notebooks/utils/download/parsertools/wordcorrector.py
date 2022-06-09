import json
import os
from pathlib import Path
from collections import defaultdict
from spellchecker import SpellChecker
from nltk.tokenize import WordPunctTokenizer
import re
from utils.download.parsertools.dictionary import (
    Dictionary,
    PhilosophyDictionary,
    RAEDictionary,
)
from multiprocessing import Pool


class WordCorrector:
    def __init__(self) -> None:
        self.dictionaries = [
            self.get_spellchecker(RAEDictionary()),
            self.get_spellchecker(PhilosophyDictionary()),
            SpellChecker(language="en"),
            SpellChecker(language="de"),
        ]

        dictionaries_path = (
            Path(__file__).parent.parent.parent.resolve() / "dictionaries"
        )
        self.corrections_path = dictionaries_path / "corrections.json"
        self.correction_counts_path = dictionaries_path / "correction_counts.json"
        if self.corrections_path.exists():
            with open(self.corrections_path) as fp:
                self.corrections = defaultdict(int, json.load(fp))
        else:
            self.corrections = {}

        if self.correction_counts_path.exists():
            with open(self.correction_counts_path) as fp:
                self.correction_counts = defaultdict(int, json.load(fp))
        else:
            self.correction_counts = defaultdict(int)

    def correct_token(self, token: str):
        if token in self.corrections:
            self.correction_counts[token] += 1
            return self.corrections[token]
        else:
            return token

    def correct_text(self, text: str):

        tokenized_text = WordPunctTokenizer().tokenize(text)

        unknown_words_start = self.unknown_words(tokenized_text)

        print(f"Unknown words: {len(unknown_words_start)}")

        with Pool(5) as p:
            corrected_pairs = p.map(self.find_correction, unknown_words_start)

        for word, correction in corrected_pairs:
            self.corrections[word] = correction

        tokenized_text = [self.correct_token(w) for w in tokenized_text]

        unknown_words_end = self.unknown_words(tokenized_text)
        print(f"Corrected. Final unknown words: {len(unknown_words_end)}")

        self.save_corrections()

        return (
            " ".join(tokenized_text),
            {
                "unknown_words_start": len(unknown_words_start),
                "unknown_words_end": len(unknown_words_end),
            },
        )

    def find_correction(self, word):
        if len(word) > 20:
            return word, word

        if word in self.corrections:
            correction = self.corrections[word]
        else:
            for dictionary in self.dictionaries:
                correction = dictionary.correction(word)

                if correction != word:
                    break

        print(f"{word}:{correction}")
        return word, correction

    def save_corrections(self):
        """
        Saves the dict of {word: its correction}
        under utils/dictionaries.
        """

        with open(self.corrections_path, "w") as fp:
            json.dump(self.corrections, fp)

        with open(self.correction_counts_path, "w") as fp:
            json.dump(self.correction_counts, fp)

    def check_word(self, word):
        for dictionary in self.dictionaries:
            if word in dictionary:
                return True

        return False

    def unknown_words(self, words: list):
        return set(
            [word for word in words if not self.check_word(word) and word.isalpha()]
        )

    def get_spellchecker(self, dictionary: Dictionary) -> SpellChecker:
        return SpellChecker(local_dictionary=dictionary.get_dictionary_path())

    def process_long_token(self, string: str):
        for dictionary in self.dictionaries:
            known_words = self.get_words_by_len(dictionary)

            for word in [w for w in known_words if len(w) > 5]:
                if word in string:
                    return [t for t in re.split(f"({word})", string) if t]

    def get_words_by_len(self, dictionary: SpellChecker):
        return sorted(list(dictionary.word_frequency.words()), key=len, reverse=True)


if __name__ == "__main__":
    corrector = WordCorrector()
