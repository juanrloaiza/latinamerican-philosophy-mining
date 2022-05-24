from collections import defaultdict
from dataclasses import dataclass
from utils.registry import Registry
from utils.filemanager import FileManager
import os

DATA_FOLDER = os.path.abspath("../../data")


@dataclass
class Article:
    """Implements an article interface. The minimal info it has is an ID."""

    id: str

    def get_bow_list(self) -> list:
        """Returns a bag of words in a list format from a string format."""
        try:
            bow = self.bag_of_words.split()
            return [w for w in bow if len(w) > 1]
        except AttributeError:
            return []


class Corpus:
    def __init__(self, registry_path: str) -> None:
        """Implements a Corpus class to manage the articles in the corpus."""
        registry_path = os.path.abspath(registry_path)
        self.registry = Registry(
            registry_path=registry_path, data_path=DATA_FOLDER, manager=FileManager()
        )
        self.documents = self.load_documents_from_registry()

    def load_documents_from_registry(self, only_articles: bool = True) -> list:
        """
        Returns a list of articles.
        """
        articles = self.registry.load_article_files()
        docs = []
        for info in [info for info in articles if info["lang"] == "es"]:
            if not only_articles or info["type"] == "ART\u00cdCULOS":
                article = Article(info["id"])
                for key, value in info.items():
                    setattr(article, key, value)
                docs.append(article)
        return docs

    def get_documents_list(self) -> list:
        results = [doc for doc in self.documents if doc.text]
        print(f"Loading corpus. Num. of articles: {len(results)}")
        return results

    def save_documents(self):
        """Saves the article's dictionary through the registry after some edit."""
        for article in self.documents:
            self.registry.update_article(article.id, article.__dict__)

    def get_article_ref(self, id) -> str:
        """Returns a string with the reference of an article."""
        for article in self.documents:
            if article.id == id:
                return f"{article.author} ({article.date}). {article.title}"

    def get_article_by_id(self, id: str) -> Article:
        """Returns an Article object given an ID."""
        for doc in self.documents:
            if doc.id == id:
                return doc

    def __len__(self):
        """Represents the length of the corpus as the number of documents it has."""
        return len(self.documents)

    def get_time_slices(self):
        slices = defaultdict(int)
        for article in self.documents:
            year = int(article.date[:4])
            slices[year] += 1

        return [count for year, count in sorted(slices.items())]
