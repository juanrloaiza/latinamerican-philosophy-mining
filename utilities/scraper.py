#!/usr/bin/env python
# coding: utf-8

"""
This script downloads articles from Ideas y Valores
and saves them in raw HTML. It also retrieves their metadata
and saves them in JSON format.

Each article will be saved in the folder data/raw_html/{articleID}/.
"""

import requests  # Allows us to make web requests
from bs4 import BeautifulSoup  # Parses HTML
import os  # To save HTML files
from time import sleep  # To wait between requests
import json  # We save metadata in JSON format
import re
import lxml

# Headers for requests
headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:76.0) Gecko/20100101 Firefox/76.0',
}

# Years to scrape (initially 2009 - 2017)
yearStart = 2009
yearEnd = 2017

# We look at the archive and parse it with BeautifulSoup
urlArchive = 'https://revistas.unal.edu.co/index.php/idval/issue/archive'
htmlArchive = requests.get(urlArchive).content
soupArchive = BeautifulSoup(htmlArchive, features = 'lxml')

# We make a list to save all issue links
issueLinks = []

# For each year we save the link each issue's table of contents.
for year in range(yearStart, yearEnd + 1):
    # Issue links are in h4 tags
    # We look at each h4 and if it contains the year we are looking for,
    # we save it.
    # We append '/showToc' to the URL.
    issueLinks += [link.a['href']+'/showToc' for link in soupArchive.find_all('h4') if str(year) in link.text]

# We visit each issue and download every article in HTML.
for issueURL in issueLinks:

    issueHTML = requests.get(issueURL, headers=headers).content
    soup = BeautifulSoup(issueHTML, features = 'lxml')
    articleLinks = [link for link in soup.find_all('a', href=True) if link.text == 'HTML']

    # Which issue are we visiting?
    print(soup.title.text)

    # We visit each article in each issue.
    for link in articleLinks:

        articleHTML = requests.get(link['href']).content
        soup = BeautifulSoup(articleHTML, features = 'lxml')

        # Which article are we visiting?
        print(f'\t{soup.title.text}')

        # We extract the metadata and pass it to a dictionary.
        meta = soup.find_all('meta', attrs={'name': True})

        metaDict = {}
        for item in meta:
            metaDict[item['name']] = item['content']

        # We extract the article ID to use it as a filename later.
        articleID = metaDict['DC.Identifier']

        # We visit printer friendly versions which are easier to parse.
        printerURL = re.findall('\'(.*)\'', soup.find(text=re.compile('Imprima')).parent['href'])[0]

        printerfriendlyHTML = requests.get(printerURL, headers=headers).content
        soup = BeautifulSoup(printerfriendlyHTML, features = 'lxml')
        content = soup.find('div', id = 'content')

        # We create a folder for each article.
        folder = f'../data/raw_html/{articleID}/'

        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(f"{folder}/{articleID}.html", 'w') as f:
            f.write(str(content))

        with open(f"{folder}/{articleID}.json", 'w') as f:
            json.dump(metaDict, f)

        # We wait five seconds before our next requests.
        sleep(5)
