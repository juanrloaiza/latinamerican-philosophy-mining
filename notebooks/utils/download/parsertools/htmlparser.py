from utils.download.parsertools.dictionary import PhilosophyDictionary


class HTMLParser:
    def __init__(self, dictionary_path: str) -> None:
        self.dictionary_path = dictionary_path

    def parse(self, html_file):
        with open(html_file) as file:
            text = file.read().strip()

        """
        Send the text to the dictionary for analysis.
        We want to build a word frequency list out of our HTML files
        in order to help spell checking the PDF files. This is because
        a lot of the vocabulary is technical, and hence needs a specialized
        list to correct words accurately. 
        """

        PhilosophyDictionary(self.dictionary_path).take_text(text)

        return text, {}
