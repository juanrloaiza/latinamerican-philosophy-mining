from utils.download.scraper import Scraper
from utils.download.articleparser import ArticleParser
from utils.filemanager import FileManager
from utils.registry import Registry
import os

ARCHIVE_URL = "https://revistas.unal.edu.co/index.php/idval/issue/archive"
DATA_FOLDER = os.path.abspath("../data")
REGISTRY_FILE = "utils/article_registry.json"
DICTIONARIES = "utils/dictionaries"

if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)

registry = Registry(
    registry_path=REGISTRY_FILE, data_path=DATA_FOLDER, manager=FileManager()
)

# Loads the current files from the data folder.
# registry.load_folder()

# Downloads the articles.
# scraper = Scraper(archive_url=ARCHIVE_URL, registry=registry)
# scraper.scrape()
# print("Downloaded all articles! Now let's get parsing...")

# Parses the downloaded articles.
parser = ArticleParser(registry=registry, dictionary_path=DICTIONARIES)
parser.parse()
