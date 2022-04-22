from dataclasses import dataclass
from registry import Registry
from filemanager import FileManager
import os

DATA_FOLDER = os.path.abspath("../data")


class Corpus:
    def __init__(self, registry_path = 'article_registry.json') -> None:
        self.registry = Registry(
            registry_path=registry_path, data_path=DATA_FOLDER, manager=FileManager()
        )
        self.documents = self.load_documents()  # {id: info}

    def load_documents(self):
        articles = self.registry.load_article_files()
        docs = []
        for info in articles:
            article = Article(info["id"])
            for key, value in info.items():
                setattr(article, key, value)
            docs.append(article)
        return docs

    def get_documents_list(self, lang="es", type="ARTÍCULOS"):
        results = [doc for doc in self.documents if doc.text]
        if lang:
            results = [doc for doc in results if doc.lang == lang]
        if type:
            results = [doc for doc in results if doc.lang == type]

        print(f"Loading corpus. Num. of articles: {len(results)}")
        return results

    def save_documents(self):
        for article in self.documents:
            self.registry.update_article(article.id, article.__dict__)

    def __len__(self):
        return len(self.documents)


@dataclass
class Article:
    id: int
