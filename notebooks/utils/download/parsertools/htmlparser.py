from utils.download.parsertools.dictionary import PhilosophyDictionary


class HTMLParser:
    def __init__(self) -> None:
        pass

    def parse(self, html_file):
        """Parses the HTML files and returns text."""

        with open(html_file, encoding="utf-8") as file:
            text = file.read().strip()

        PhilosophyDictionary().take_text(text)

        return text, {}
