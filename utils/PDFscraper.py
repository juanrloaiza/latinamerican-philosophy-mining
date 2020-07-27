#!/usr/bin/env python
# coding: utf-8

"""
This script downloads articles from Ideas y Valores
and saves them in PDF. It also retrieves their metadata
and saves them in JSON format.

It uses the HTML scraper as a base.

Each article will be saved in the folder data/rawPDF/{articleID}/.
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

# We will skip years from 2009-2017, since we get these articles in HTML format.
yearToSkip = [x for x in range(2009, 2018)]

# There are 3 pages containing links to the 137 issues of the journal.
# We loop through these three pages collecting links.
issueLinks = []
for x in range(4):

    # We look at the archive and parse it with BeautifulSoup
    urlArchive = f'https://revistas.unal.edu.co/index.php/idval/issue/archive?issuesPage={x}#issues'
    htmlArchive = requests.get(urlArchive).content
    soupArchive = BeautifulSoup(htmlArchive, features = 'lxml')

    # Issue links are in h4 tags
    # We look at each h4 and if it contains a year we can skip, we skip.
    # We append '/showToc' to the URL.
    for link in soupArchive.find_all('h4'):
        for year in yearToSkip:
            if str(year) not in link.text:
                issueLinks.append(link.a['href']+'/showToc')

issueLinks = set(issueLinks) # There are a ton of duplicates with the current implementation.

# We visit each issue and download every article in PDF.
for issueURL in issueLinks:

    issueHTML = requests.get(issueURL, headers=headers).content
    soup = BeautifulSoup(issueHTML, features = 'lxml')
    articleLinks = [link for link in soup.find_all('a', href=True) if link.text == 'HTML']

    # Which issue are we visiting?
    print(soup.title.text)

    # We visit each article in each issue.
    for link in articleLinks:

        # We get the PDF
        articlePDF = requests.get(link['href'].replace('view', 'download')).content

        # We also visit the HTML to get the metadata.
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
        folder = f'../data/rawPDF/{articleID}/'

        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(f"{folder}/{articleID}.pdf", 'w') as f:
            f.write(articlePDF)

        with open(f"{folder}/{articleID}.json", 'w') as f:
            json.dump(metaDict, f)

        # We wait five seconds before our next requests.
        sleep(5)
