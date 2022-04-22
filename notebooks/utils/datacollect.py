from datacollection.scraper import Scraper
from datacollection.articleparser import ArticleReader
from filemanager import FileManager
from registry import Registry
import os

ARCHIVE_URL = "https://revistas.unal.edu.co/index.php/idval/issue/archive"
DATA_FOLDER = "../data"
REGISTRY_FILE = "article_registry.json"
DICTIONARIES = "./dictionaries"

if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)

registry = Registry(
    registry_path=REGISTRY_FILE, data_path=DATA_FOLDER, manager=FileManager()
)

scraper = Scraper(archive_url=ARCHIVE_URL, registry=registry)
scraper.scrape()

parser = ArticleReader(registry=registry, dictionary_path=DICTIONARIES)
parser.parse()
