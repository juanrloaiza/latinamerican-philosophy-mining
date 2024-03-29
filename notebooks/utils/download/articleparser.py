from utils.download.parsertools.pdfparser import PDFParser
from utils.download.parsertools.htmlparser import HTMLParser
from utils.download.parsertools.jsonparser import JSONParser
from utils.registry import Registry, Format


class ArticleParser:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry

    def parse(self):
        article_id_list = self.registry.get_article_id_list()

        html_articles = [
            article_id
            for article_id in article_id_list
            if self.registry.get_article_format(article_id) == Format.HTML.value
        ]

        for article in html_articles:
            self.parse_article(article, HTMLParser())

        pdf_articles = [
            article_id
            for article_id in article_id_list
            if self.registry.get_article_format(article_id) == Format.PDF.value
        ]

        for article in pdf_articles:
            self.parse_article(
                article,
                PDFParser(registry=self.registry),
            )

    def parse_article(self, article_id, articleparser):
        if self.registry.is_article_parsed(article_id):
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
