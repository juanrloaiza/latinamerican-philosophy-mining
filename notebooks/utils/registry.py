import json
import os
from enum import Enum

from filemanager import FileManager


class Format(Enum):
    """Possible formats a file can have."""

    PDF: str = "pdf"
    HTML: str = "html"
    JSON: str = "json"


class Registry:
    def __init__(
        self, registry_path: str, data_path: str, manager: FileManager,
    ):
        """
        TODO: Change the paths to a dictionary we can read from JSON.
        """
        self.registry_path = registry_path
        self.data_path = data_path
        self.manager = manager

        # Check if registry exists. If not, create it.
        if not os.path.exists(registry_path):
            self.create_registry()

        self.database = self.load_registry()

    def create_registry(self):
        with open(self.registry_path, "w") as file:
            json.dump({}, file)

    def save_registry(self):
        with open(self.registry_path, "w") as file:
            json.dump(self.database, file, indent=True)

    def load_registry(self):
        print(self.registry_path)
        with open(self.registry_path) as file:
            return json.load(file)

    def add_article(
        self, id: int, format: Format, raw_content: bytes, raw_metadata: dict
    ):
        """Registers and saves a downloaded article."""

        # Define the information to store in the database
        folder = f"{self.data_path}/raw/{id}"
        raw_filepath = f"{folder}/{id}.{format.value}"

        article_dict = {
            "id": id,
            "raw_folder": f"{folder}",
            "raw_metadata": f"{folder}/{id}.json",
            "raw_filepath": raw_filepath,
            "format": format.value,
            "parsed": False,
        }

        # Save the file
        self.manager.save_raw_data(
            id, format.value, raw_content=raw_content, raw_folder=folder
        )
        self.manager.save_raw_data(
            id, "json", raw_content=raw_metadata, raw_folder=folder
        )

        # Register the file into database.
        self.database[id] = article_dict
        self.save_registry()

    def update_article(self, id: str, new_info: dict):
        for key, value in new_info.items():
            self.database[id][key] = value

        self.manager.save(self.database[id], self.database[id]["filepath"])
        self.save_registry()

    def get_article_raw_file(self, id: str):
        return self.database[id]["raw_filepath"]

    def get_article_format(self, id: str):
        return self.database[id]["format"]

    def get_article_raw_metadata(self, id: str):
        return self.database[id]["raw_metadata"]

    def get_article_id_list(self):
        return self.database.keys()

    def load_article_files(self):
        articles = []
        for info in self.database.values():
            articles.append(info)
        print(len(articles))
        return [self.manager.load(article["filepath"]) for article in articles]

    def check_article_parsed(self, id: str):
        return self.database[id]["parsed"]

    def check_article_downloaded(self, id: str):
        return id in self.database.keys()

    def save_parsed_article(self, id, article):
        """Registers and saves the article once it's parsed."""

        filepath = f"{self.data_path}/corpus/{id}.json"

        self.manager.save(article, filepath)

        self.database[id]["parsed"] = True
        self.database[id]["filepath"] = filepath

        self.save_registry()

