from datacollection.parsertools.pdfparser import PDFParser
from datacollection.parsertools.htmlparser import HTMLParser
from datacollection.parsertools.jsonparser import JSONParser
from registry import Registry, Format


class ArticleReader:
    def __init__(self, registry: Registry, dictionary_path: str) -> None:
        self.registry = registry
        self.dictionary_path = dictionary_path

    def parse(self):
        article_id_list = self.registry.get_article_id_list()

        html_articles = [
            article_id
            for article_id in article_id_list
            if self.registry.get_article_format(article_id) == Format.HTML.value
        ]

        for article in html_articles:
            self.parse_article(
                article, HTMLParser(dictionary_path=self.dictionary_path)
            )

        pdf_articles = [
            article_id
            for article_id in article_id_list
            if self.registry.get_article_format(article_id) == Format.PDF.value
        ]

        for article in pdf_articles:
            self.parse_article(
                article,
                PDFParser(dictionary_path=self.dictionary_path, registry=self.registry),
            )

    def parse_article(self, article_id, articleparser):
        # Check if we already parsed this article.
        if self.registry.check_article_parsed(article_id):
            print("Already parsed.")
            return

        raw_metadata = self.registry.get_article_raw_metadata(article_id)
        article_file = self.registry.get_article_raw_file(article_id)

        print(f"Parsing: {article_file}")

        article_metadata = JSONParser().parse(raw_metadata)

        article_text, corrector_results = articleparser.parse(article_file)

        article_metadata["text"] = article_text
        for key, value in corrector_results.items():
            article_metadata[key] = value

        self.registry.save_parsed_article(article_id, article_metadata)
