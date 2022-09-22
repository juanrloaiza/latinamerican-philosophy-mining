from dataclasses import dataclass
from pathlib import Path
from utils.registry import Registry
from utils.filemanager import FileManager


DATA_FOLDER = Path(__file__).parent.parent.parent.resolve() / "data"


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

    def get_ref(self) -> str:
        """Returns the article's reference."""
        return f"{self.author} ({self.date}). {self.title}"


class Corpus:
    def __init__(self, registry_path: str) -> None:
        """Implements a Corpus class to manage the articles in the corpus."""
        registry_path = Path(registry_path).resolve()

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
            if (not only_articles or info["type"] == "ART\u00cdCULOS") and info["text"] != "":
                article = Article(info["id"])
                for key, value in info.items():
                    setattr(article, key, value)
                docs.append(article)

        docs = sorted(docs, key=lambda article: int(article.date[:4]))
        return docs

    def get_documents_list(self) -> list:
        """Gets a list of documents that have a text ready for preprocessing."""
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

    def get_time_slices(self, time_window: int = 5):
        """
        Returns time slices for LDA Sequential model.

        Time slices are a list of how many articles a given time window has. This is
        a requirement of LDASeqModel.
        """
        all_years = [int(article.date[:4]) for article in self.documents]
        bins = (all_years[-1] - 1950) // time_window + 1

        count_by_year = {year: all_years.count(year) for year in all_years}

        self.time_slice_years = []

        current_year = 1950
        next_year = current_year + time_window

        counts = []
        for _ in range(bins):
            count = sum(
                [v for k, v in count_by_year.items() if current_year <= k < next_year]
            )
            counts.append(count)

            print(f"{current_year} - {next_year - 1}: {count}")
            self.time_slice_years.append((current_year, next_year))
            current_year, next_year = next_year, next_year + time_window

        print(self.time_slice_years)

        return counts


if __name__ == "__main__":
    print("Data folder:", DATA_FOLDER)

    corpus = Corpus(registry_path="notebooks/utils/article_registry.json")

    print("Testing time slices:")
    corpus.get_time_slices(time_window=50)
