import json


class JSONParser:
    def __init__(self) -> None:
        pass

    def parse(self, json_file):
        with open(json_file) as file:
            raw_metadata = json.load(file)

        new_metadata = {
            "id": raw_metadata["DC.Identifier"],
            "lang": raw_metadata["citation_language"],
            "author": raw_metadata["citation_author"],
            "date": raw_metadata["citation_date"],
            "type": raw_metadata["DC.Type.articleType"],
        }

        try:
            new_metadata["vol"] = raw_metadata["citation_volume"]
        except KeyError:
            new_metadata["vol"] = "Undefined"

        try:
            new_metadata["issue"] = raw_metadata["citation_issue"]
        except KeyError:
            new_metadata["issue"] = "Undefined"

        return new_metadata
