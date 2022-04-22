import os
import json


class FileManager:
    def __init__(self) -> None:
        pass

    def load(self, path):
        with open(path) as file:
            return json.load(file)

    def update(self, path, new_info: dict):
        article = self.load(path)

        for key, value in new_info.items():
            article[key] = value

        self.save(article, path)

    def save(self, article, path):
        folder = os.path.dirname(path)
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(path, "w") as file:
            json.dump(article, file, indent=True)

    def save_raw_data(self, id, format, raw_content, raw_folder):
        raw_filepath = f"{raw_folder}/{id}.{format}"

        if not os.path.exists(raw_folder):
            os.makedirs(raw_folder)

        if format == "pdf":
            with open(f"{raw_filepath}", "wb") as f:
                f.write(raw_content)
        elif format == "html":
            with open(f"{raw_filepath}", "w") as f:
                f.write(str(raw_content))
        elif format == "json":
            with open(f"{raw_folder}/{id}.json", "w", encoding="utf-8") as f:
                json.dump(raw_content, f, indent=True)

