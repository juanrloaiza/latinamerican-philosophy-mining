import json
import os
from enum import Enum

from utils.filemanager import FileManager


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
        with open(self.registry_path) as file:
            return json.load(file)

    def add_article(
        self, id: int, format: Format, raw_content: bytes, raw_metadata: dict, url: str
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
            "url": url,
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
        paths = []
        for info in self.database.values():
            if "filepath" in info.keys():
                paths.append(info["filepath"])
        return [self.manager.load(path) for path in paths]

    def check_article_parsed(self, id: str):
        return self.database[id]["parsed"]

    def check_article_downloaded(self, url: str):
        for info in self.database.values():
            if info["url"] in url:
                return True
        return False

    def save_parsed_article(self, id, article):
        """Registers and saves the article once it's parsed."""

        filepath = f"{self.data_path}/corpus/{id}.json"

        self.manager.save(article, filepath)

        self.database[id]["parsed"] = True
        self.database[id]["filepath"] = filepath

        self.save_registry()

    def load_folder(self):
        for id in os.listdir("../data/raw"):
            raw_folder = os.path.abspath(f"../data/raw/{id}")
            file1, file2 = os.listdir(raw_folder)
            if "json" in file1:
                raw_metadata = f"{raw_folder}/{file1}"
                raw_filepath = f"{raw_folder}/{file2}"
            else:
                raw_filepath = f"{raw_folder}/{file1}"
                raw_metadata = f"{raw_folder}/{file2}"
            if "pdf" in raw_filepath:
                format = "pdf"
            else:
                format = "html"
            with open(raw_metadata) as file:
                info = json.load(file)
            url = info["DC.Identifier.URI"]

            article_dict = {
                "id": id,
                "raw_folder": f"{raw_folder}",
                "raw_metadata": f"{raw_folder}/{id}.json",
                "raw_filepath": raw_filepath,
                "format": format,
                "url": url,
                "parsed": False,
            }

            # Register the file into database.
            self.database[id] = article_dict
            self.save_registry()
            print(f"Added {id} from folder.")
