# coding: utf-8

"""
This script downloads articles from Ideas y Valores
and saves them in raw HTML. It also retrieves their metadata
and saves them in JSON format.

Each article will be saved in the folder data/raw_html/{article_id}/.
"""

from datacollection.scrapertools.downloaders import (
    HTMLDownloader,
    PDFDownloader,
    soupify,
)
from registry import Registry


# Headers for requests
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:76.0) Gecko/20100101 Firefox/76.0",
}


class Scraper:
    def __init__(
        self, archive_url: str, registry: Registry, headers: dict = DEFAULT_HEADERS,
    ):

        self.archive_urls = []
        self.headers = headers
        self.registry = registry

        self.get_next_archive_page(archive_url)

    def scrape(self):
        issue_toc_links = []

        for archive_page_url in self.archive_urls:
            issue_toc_links += self.scrape_archive_page(archive_page_url)

        # We visit each issue and download every article.
        for issue_toc in issue_toc_links[:1]:
            self.scrape_issue(issue_toc)

    def get_next_archive_page(self, current_url):
        """Recursive function to get all archive pages."""
        self.archive_urls.append(current_url)

        current_page_soup = soupify(current_url)

        if not current_page_soup.find("a", class_="next"):
            return

        self.get_next_archive_page(current_page_soup.find("a", class_="next")["href"])

    def scrape_archive_page(self, archive_page_url):
        archive_page_soup = soupify(archive_page_url)

        # For each year we save the link to each issue's table of contents.
        issue_summaries = archive_page_soup.find_all("div", class_="obj_issue_summary")

        return [div.find("a", class_="title")["href"] for div in issue_summaries]

    def scrape_issue(self, toc_link: str):

        # Visit issue HTML
        issue_soup = soupify(toc_link)

        # Get link for each article
        for article_summary in issue_soup("div", class_="obj_article_summary"):
            if link_tag := article_summary.find("a", class_="obj_galley_link html"):
                downloader = HTMLDownloader(self.registry)
            elif link_tag := article_summary.find("a", class_="obj_galley_link pdf"):
                downloader = PDFDownloader(self.registry)
            else:
                print("Didn't find a suitable link. Skipping")
                continue

            downloader.download(link_tag["href"])
