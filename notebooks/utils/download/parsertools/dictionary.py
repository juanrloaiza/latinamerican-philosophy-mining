from abc import ABC, abstractmethod
import json
from pathlib import Path
from spellchecker import SpellChecker


dictionaries_folder = Path(__file__).parent.parent.parent.resolve() / "dictionaries"


class Dictionary(ABC):
    """Defines abstract dictionary functionality."""

    def __init__(self, filename) -> None:
        self.path = dictionaries_folder / filename
        self.spell = SpellChecker(language=None)

        if not self.path.exists():
            self.create_dictionary()

        self.load_dictionary()

    @abstractmethod
    def create_dictionary(self) -> None:
        """Creates the dictionary file."""

    def load_dictionary(self) -> None:
        """Loads the dictionary from file"""
        self.spell.word_frequency.load_dictionary(str(self.path))

    def get_dictionary_path(self) -> str:
        """Returns the path to the JSON dictionary file."""
        return str(self.path)


class PhilosophyDictionary(Dictionary):
    """Dictionary based on the HTML files from Ideas y Valores."""

    def __init__(self, filename="philosophy_dictionary.json"):
        Dictionary.__init__(self, filename)

    def create_dictionary(self) -> None:
        with open(self.path, "w", encoding="utf-8") as file:
            json.dump({}, file)

    def take_text(self, text) -> None:
        """
        Takes in text and adds frequencies to the current dictionary in memory.
        Then it saves the frequencies to file.
        """

        words = text.split()
        self.spell.word_frequency.load_words(
            [w for w in words if w.isalpha() and len(w) > 3]
        )
        self.spell.word_frequency.remove_by_threshold(3)
        self.spell.export(self.path, gzipped=False)


class RAEDictionary(Dictionary):
    """Dictionary based on the corpus by the Real Academia Española (RAE)."""

    def __init__(self, filename="rae_processed.json"):
        Dictionary.__init__(self, filename)

    def create_dictionary(self) -> None:
        """Creates a RAE JSON dictionary from the text file downloaded from the RAE."""
        # TODO: Download the file in the code. Involves unzipping the file.

        raw_filepath = dictionaries_folder / "rae_original.TXT"

        print("Creating RAE Dictionary.")
        with open(raw_filepath, encoding="iso8859-15") as file:
            raw_rae_dict = file.read()

        rae_dict = {}
        for line in raw_rae_dict.split("\n")[1:]:
            if len(line.split("\t")) == 4:
                number, word, frequency, rel_frequency = line.split("\t")
                frequency = int(frequency.replace(",", ""))
                rae_dict[word.strip()] = frequency

        self.spell.word_frequency.load_json(rae_dict)
        self.spell.word_frequency.remove_by_threshold(100)
        self.spell.export(self.path, gzipped=False)


if __name__ == "__main__":
    phil_dictionary = PhilosophyDictionary()
    rae_dictionary = RAEDictionary()

    print(phil_dictionary.spell.correction("analitico"))
    print(rae_dictionary.spell.correction("analitico"))
