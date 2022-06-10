import json
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
from spellchecker import SpellChecker
from nltk.tokenize import WordPunctTokenizer
from utils.download.parsertools.dictionary import PhilosophyDictionary, RAEDictionary


class WordCorrector:
    def __init__(self) -> None:
        self.dictionaries = [
            SpellChecker(local_dictionary=RAEDictionary().get_dictionary_path()),
            SpellChecker(local_dictionary=PhilosophyDictionary().get_dictionary_path()),
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

    def correct_token(self, token: str) -> str:
        """Takes a token and returns its replacement."""
        if token in self.corrections:
            self.correction_counts[token] += 1
            return self.corrections[token]
        else:
            return token

    def correct_text(self, text: str) -> tuple[str, dict]:
        """Takes a whole text and returns a corrected version, along with a dictionary
        with how many words did we not recognize at first vs. how many words are recognized
        after correcting the text."""

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

    def find_correction(self, word: str) -> tuple[str, str]:
        """Finds the correction for a word in the dictionaries and returns it with its correction.
        If there is no different word for a correction, return the same word back."""

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

    def save_corrections(self) -> None:
        """
        Saves the dict of {word: its correction}
        under utils/dictionaries.
        """

        with open(self.corrections_path, "w") as fp:
            json.dump(self.corrections, fp, indent=True)

        with open(self.correction_counts_path, "w") as fp:
            json.dump(self.correction_counts, fp, indent=True)

    def check_word(self, word) -> bool:
        """Checks if we know a word among all dictionaries available."""
        for dictionary in self.dictionaries:
            if word in dictionary:
                return True

        return False

    def unknown_words(self, words: list) -> set:
        """Returns a set of unknown words (alphanumeric strings) from a list among all dictionaries."""
        return set(
            [word for word in words if not self.check_word(word) and word.isalpha()]
        )


if __name__ == "__main__":
    corrector = WordCorrector()
