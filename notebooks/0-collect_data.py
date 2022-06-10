from utils.download.scraper import Scraper
from utils.download.articleparser import ArticleParser
from utils.filemanager import FileManager
from utils.registry import Registry
from pathlib import Path

ARCHIVE_URL = "https://revistas.unal.edu.co/index.php/idval/issue/archive"
data_folder = Path(__file__).parent.parent.resolve() / "data"
registry_file = Path(__file__).parent.resolve() / "utils" / "article_registry.json"

data_folder.mkdir(exist_ok=True)

registry = Registry(
    registry_path=registry_file, data_path=data_folder, manager=FileManager()
)

# Loads the current files from the data folder.
# registry.load_raw_folder()

# Downloads the articles.
# scraper = Scraper(archive_url=ARCHIVE_URL, registry=registry)
# scraper.scrape()
# print("Downloaded all articles! Now let's get parsing...")

# Parses the downloaded articles.
parser = ArticleParser(registry=registry)
parser.parse()
