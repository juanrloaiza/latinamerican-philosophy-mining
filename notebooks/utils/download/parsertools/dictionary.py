from abc import ABC, abstractmethod
import json
import os
from spellchecker import SpellChecker


class Dictionary(ABC):
    def __init__(self, base_path) -> None:
        """
        Implements abstract dictionary functionality.
        """

    @abstractmethod
    def create_dictionary(self, path):
        """Creates the dictionary in a path."""

    def load_dictionary(self):
        self.spell.word_frequency.load_dictionary(self.path)

    def get_dictionary_path(self) -> str:
        """Returns the path to the RAE JSON dictionary."""
        return self.path


class PhilosophyDictionary(Dictionary):
    def __init__(self, base_path) -> None:
        self.path = f"{base_path}/philosophy_dictionary.json"
        self.spell = SpellChecker(language=None)

        if not os.path.exists(self.path):
            self.create_dictionary()

        self.load_dictionary()

    def create_dictionary(self) -> None:
        with open(self.path, "w") as file:
            json.dump({}, file)

    def take_text(self, text) -> None:
        """
        Takes in text and adds frequencies to the current dictionary in memory.
        Then it saves the frequencies to file.
        """
        words = text.split()
        self.spell.word_frequency.load_words([w for w in words if w.isalpha() and len(w) > 3])
        self.spell.word_frequency.remove_by_threshold(3)
        self.spell.export(self.path, gzipped=False)


class RAEDictionary(Dictionary):
    def __init__(self, base_path) -> None:
        raw_file = f"{base_path}/rae_original.TXT"
        self.path = f"{base_path}/rae_processed.json"
        self.spell = SpellChecker(language=None)

        # Check if RAE dictionary exists, otherwise create it.
        if not os.path.exists(self.path):
            self.create_dictionary(raw_file)

        self.load_dictionary()

    def create_dictionary(self, raw_file) -> None:
        """Creates a RAE JSON dictionary from the text file downloaded from the RAE."""
        # TODO: Download the file in the code. Involves unzipping the file.

        print("Creating RAE Dictionary.")
        with open(raw_file, encoding="iso8859-15") as file:
            raw_rae_dict = file.read()

        rae_dict = {}
        for line in raw_rae_dict.split("\n")[1:]:
            if len(line.split("\t")) == 4:
                number, word, frequency, rel_frequency = line.split("\t")
                frequency = int(frequency.replace(",", ""))
                rae_dict[word.strip()] = frequency
                    

        self.spell.word_frequency.load_json(rae_dict)
        self.spell.word_frequency.remove_by_threshold(5)
        self.spell.export(self.path, gzipped=False)

