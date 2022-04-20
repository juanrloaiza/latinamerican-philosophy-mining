from spellchecker import SpellChecker
from nltk.tokenize import WordPunctTokenizer
import re
from datacollection.parsertools.dictionary import (
    Dictionary,
    PhilosophyDictionary,
    RAEDictionary,
)
from multiprocessing import Pool


class WordCorrector:
    def __init__(self, dictionary_path: str) -> None:
        self.dictionaries = [
            self.get_spellchecker(RAEDictionary(dictionary_path)),
            self.get_spellchecker(PhilosophyDictionary(dictionary_path)),
            SpellChecker(language="en"),
            SpellChecker(language="de"),
        ]

    def correct_text(self, text: str):

        tokenized_text = WordPunctTokenizer().tokenize(text)

        unknown_words_start = self.unknown_words(tokenized_text)

        print(f"Unknown words: {len(unknown_words_start)}")

        with Pool() as p:
            corrected_pairs = p.map(self.correct_word, unknown_words_start)

        correction_dict = {}
        for word, correction in corrected_pairs:
            correction_dict[word] = correction

        tokenized_text = [
            correction_dict[w] if w in correction_dict.keys() else w
            for w in tokenized_text
        ]

        unknown_words_end = self.unknown_words(tokenized_text)
        print(f"Corrected. Final unknown words: {len(unknown_words_end)}")

        return (
            " ".join(tokenized_text),
            {
                "unknown_words_start": len(unknown_words_start),
                "unknown_words_end": len(unknown_words_end),
            },
        )

    def correct_word(self, word):

        for dictionary in self.dictionaries:
            correction = dictionary.correction(word)

            if correction != word:
                break

        print(f"{word}:{correction}")
        return word, correction

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
