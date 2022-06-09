import json
from enum import Enum
from pathlib import Path

from utils.filemanager import FileManager


class Format(Enum):
    """Possible formats a file can have."""

    PDF: str = "pdf"
    HTML: str = "html"
    JSON: str = "json"


class Registry:
    def __init__(
        self,
        registry_path: str,
        data_path: str,
        manager: FileManager,
    ):
        """
        The Registry object will manage a JSON file that keeps track of the documents
        we've downloaded, parsed, and where they are stored. It interfaces with a
        FileManager object to save the files to disk, and it is the source of information
        for the Corpus object to load the JSON files via the Registry.

        TODO: Change the paths to a dictionary we can read from JSON.
        """
        self.registry_path = Path(registry_path)
        self.data_path = Path(data_path)
        self.manager = manager

        # Check if registry exists. If not, create it.
        if not self.registry_path.exists():
            with open(self.registry_path, "w") as file:
                json.dump({}, file)

        self.database = self.load_registry()

    def save_registry(self):
        """Saves the registry JSON file."""
        with open(self.registry_path, "w") as file:
            json.dump(self.database, file, indent=True)

    def load_registry(self):
        """Loads the registry from JSON file."""
        with open(self.registry_path) as file:
            return json.load(file)

    def add_article(
        self, id: int, format: Format, raw_content: bytes, raw_metadata: dict, url: str
    ):
        """Registers and saves a downloaded article."""

        # Define the information to store in the database
        folder = self.data_path / "raw" / id
        raw_filepath = folder / f"{id}.{format.value}"

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

    def get_article_raw_file(self, id: str):
        """Returns raw HTML or PDF file path from a given article ID."""
        return self.database[id]["raw_filepath"]

    def get_article_format(self, id: str):
        """Returns the format of the article from ID."""
        return self.database[id]["format"]

    def get_article_raw_metadata(self, id: str):
        """Returns the JSON metadata filepath for a given article ID."""
        return self.database[id]["raw_metadata"]

    def get_article_id_list(self):
        """Returns all the ID numbers in the database."""
        return self.database.keys()

    def load_article_files(self):
        """Loads the article files from the database."""
        paths = []
        for info in self.database.values():
            if "filepath" in info:
                paths.append(info["filepath"])
        return [self.manager.load(path) for path in paths]

    def is_article_parsed(self, id: str):
        """Checks if an article has been parsed already."""
        return self.database[id]["parsed"]

    def check_article_downloaded(self, url: str):
        """Checks if the article has been downloaded already."""
        for info in self.database.values():
            if info["url"] in url:
                return True
        return False

    def save_parsed_article(self, id, article):
        """Registers and saves the article once it's parsed."""

        filepath = self.data_path / "corpus" / f"{id}.json"

        self.manager.save(article, filepath)

        self.database[id]["parsed"] = True
        self.database[id]["filepath"] = filepath

        self.save_registry()

    def load_raw_folder(self):
        """Loads articles from data/raw to database."""
        for raw_folder in (self.data_path / "raw").iterdir():

            id = raw_folder.name

            raw_metadata = raw_folder / f"{id}.json"

            raw_filepath = raw_folder / id

            if raw_filepath.with_suffix(".html").exists():
                raw_filepath = raw_filepath.with_suffix(".html")
                format = "html"
            else:
                raw_filepath = raw_filepath.with_suffix(".pdf")
                format = "pdf"

            info = self.manager.load(raw_metadata)
            url = info["DC.Identifier.URI"]
            parsed = False
            filepath = None

            if Path.exists(self.data_path / "corpus" / f"{id}.json"):
                parsed = True
                filepath = self.data_path / "corpus" / f"{id}.json"

            article_dict = {
                "id": id,
                "raw_folder": str(raw_folder),
                "raw_metadata": str(raw_metadata),
                "raw_filepath": str(raw_filepath),
                "filepath": str(filepath),
                "format": format,
                "url": url,
                "parsed": parsed,
            }

            # Register the file into database.
            self.database[id] = article_dict
            print(f"Loaded {id} from raw folder.")

        self.save_registry()


if __name__ == "__main__":

    DATA_FOLDER = Path(__file__).parent.parent.parent.resolve() / "data"
    REGISTRY_FILE = Path(__file__).parent.resolve() / "dummy_registry.json"

    registry = registry = Registry(
        registry_path=REGISTRY_FILE, data_path=DATA_FOLDER, manager=FileManager()
    )

    registry.load_raw_folder()
