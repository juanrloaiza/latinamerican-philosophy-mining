# Scraping and downloading data

To collect data, we must scrape the journal's website. To do this and keep the code relatively maintainable, we implement various classes that will do the job. The idea is to define three types of classes: 

* `Scraper`: Visits the journal's website and gets links to download files. The `Scraper` uses `Downloader` classes to implement different strategies to download files.
* `Downloader`: Interfaces for the `Scraper` class that download files and metadata given a link. These are divided into two strategies:
    * `PDFDownloader`: Downloads PDF files.
    * `HTMLDownloader`: Downloads HTML files.
* `Parser`: Reads downloaded PDF or HTML files, as well as JSON metadata files, and creates `Article` objects to start processing. This is also an interface which implements three parsing strategies:
    * `PDFParser`: Parses PDF files. Requires spell checking, which is why it calls the `WordCorrector` class.
    * `HTMLParser`: Parses HTML files.
    * `JSONParser`: Parses JSON raw metadata files to get the information that we need for analysis.
* `Registry`: Keeps track of downloaded files. This class is common to all others, since various tasks require that we look up the registry and see if we have some file, in which format did we download it, and getting its metadata.

This structure allows us to change scraping or downloading strategies more easily in the future.