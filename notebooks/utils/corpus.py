from dataclasses import dataclass
from utils.registry import Registry
from utils.filemanager import FileManager
import os

DATA_FOLDER = os.path.abspath("../../data")


class Corpus:
    def __init__(self, registry_path: str) -> None:
        registry_path = os.path.abspath(registry_path)
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

    def get_documents_list(self, lang="es", type="ART\u00cdCULOS") -> list:
        results = [doc for doc in self.documents if doc.text]
        if lang:
            results = [doc for doc in results if doc.lang == lang]
        if type:
            results = [doc for doc in results if doc.type == type]

        print(f"Loading corpus. Num. of articles: {len(results)}")
        return results

    def get_article_ref(self, id):
        for article in self.documents:
            if article.id == id:
                return f"{article.author} ({article.date}). {article.title}"

    def save_documents(self):
        for article in self.documents:
            self.registry.update_article(article.id, article.__dict__)

    def get_article_by_id(self, id: str):
        for doc in self.documents:
            if doc.id == id:
                return doc

    def __len__(self):
        return len(self.documents)


@dataclass
class Article:
    id: int

    def get_bag_of_words(self) -> list:
        try:
            bow = self.bag_of_words.split()
            return [w for w in bow if len(w) > 1]
        except AttributeError:
            return []
