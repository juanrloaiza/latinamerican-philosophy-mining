"""
Defines the Article class which allows us more readable
code and easy metadata parsing.

ToDo:
    * Document the class
"""
from bs4 import BeautifulSoup
import re

class Article:

    def __init__(self, html, meta, isHTML = False):
        if isHTML:
            self.parseHTML(html, meta)

    def parseHTML(self, html, meta):
        soup = BeautifulSoup(html)

        self.id = meta['DC.Identifier']
        self.title = meta['description']
        self.author = meta['DC.Creator.PersonalName']
        self.language = meta['DC.Language']
        self.issue = meta['DC.Source.Issue']
        self.vol = meta['DC.Source.Volume']
        self.date = meta['citation_date']
        self.type = meta['DC.Type.articleType']

        # Sorprendentemente, no todos tienen keywords
        # self.keywords = meta['keywords'].split('; ')

        text = soup.text
        if 'Bibliografa' in text:
            text = re.findall('.*(?=Bibliografía)', text, re.DOTALL)[0]
            self.bib = soup.find(text=re.compile('Bibliografía')).find_all_next()

        if 'Keywords' in text:
            text = text.split(re.findall('(Keywords:?\n?.*)', text)[0], 1)[1]

        self.text = text

    # Representación en string de cada artículo, por si acaso.
    def __str__(self):
        return "{} ({}). {}. {}({})".format(self.author, self.date, self.title, self.vol, self.issue)
