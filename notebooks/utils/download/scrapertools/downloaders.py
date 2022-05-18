"""Implements downloader classes to get files from the website."""

import requests
from bs4 import BeautifulSoup
from abc import abstractmethod
from utils.registry import Registry, Format

"""UTILITY FUNCTIONS"""


def soupify(url: str):
    """Returns a BeautifulSoup object from a url."""
    return BeautifulSoup(requests.get(url).content, features="lxml")


def extract_metadata(soup: BeautifulSoup):
    """
    Extracts metadata from a BeautifulSoup object. 
    
    Returns a tuple of the article ID and its metadata.
    """

    meta = soup.find_all("meta", attrs={"name": True})

    metadata = {}
    for item in meta:
        metadata[item["name"]] = item["content"]

    articleID = metadata["DC.Identifier"]

    return articleID, metadata


"""MAIN CLASSES"""


class Downloader:
    """
    General downloader. It will download files given a link.
    Implements a strategy pattern with an abstract downloading method.
    """

    def __init__(self, registry: Registry):
        self.registry = registry

    @abstractmethod
    def download(self, url):
        pass


class HTMLDownloader(Downloader):
    """Downloads HTML files."""

    def __init__(self, registry: Registry):
        self.registry = registry

    def download(self, article_url):

        article_soup = soupify(article_url)

        # Extract metadata
        id, raw_metadata = extract_metadata(article_soup)

        # Which article are we visiting?
        print(f"Title: {raw_metadata['DC.Title']}")
        print("Type: HTML")

        article_html_data = article_soup.find("div", class_="textoCompleto").text

        self.registry.add_article(
            id=id,
            raw_content=article_html_data,
            raw_metadata=raw_metadata,
            format=Format.HTML,
            url=article_url,
        )


class PDFDownloader(Downloader):
    """Downloads PDF files."""

    def __init__(self, registry: Registry):
        self.registry = registry

    def download(self, article_url):

        pdf_soup = soupify(article_url)

        # Issues now point directly to the pdf file.
        # We need to go back to the main article page to get the metadata.

        if not pdf_soup.find("a", class_="return"):
            print("Couldn't go back for the metadata.")
            return

        article_link = pdf_soup.find("a", class_="return")["href"]
        article_soup = soupify(article_link)

        article_id, raw_metadata = extract_metadata(article_soup)

        print(f"Title: {raw_metadata['DC.Title']}")
        print("Type: PDF")

        download_tag = pdf_soup.find("a", class_="download")

        if not download_tag:
            print("Couldn't get where to download the pdf file from!")
            pdf = None
            format = None
        else:
            pdf = requests.get(download_tag["href"]).content
            format = Format.PDF

        self.registry.add_article(
            id=article_id,
            raw_content=pdf,
            raw_metadata=raw_metadata,
            format=format,
            url=article_url,
        )
