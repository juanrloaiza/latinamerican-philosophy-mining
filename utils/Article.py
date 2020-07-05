"""
Defines the Article class which allows us more readable
code and easy metadata parsing.

ToDo:
    * Document the class
"""
from bs4 import BeautifulSoup
import re
import json

class Article:
    def __init__(self, file = None, html = None, meta = None):
        if html:
            self.parseHTML(html, meta)
        elif file:
            self.parseJSON(file)
        else:
            raise ValueError("One of 'file' or 'html' should be different to None.")

    def parseHTML(self, html, meta):
        """
        Method in case we get an html file. To be used when parsing HTML files for the first time.
        """
        soup = BeautifulSoup(html)

        self.id = meta['DC.Identifier']
        self.title = meta['description']
        self.author = meta['DC.Creator.PersonalName']
        self.lang = meta['DC.Language']
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

    def parseJSON(self, file):
        """
        Method in case we get a JSON file.
        We load it and then we process it simply.

        TODO: There's a way of just storing all
        keys from a dict as attributes of the
        class. Should we go for that?
        """
        meta = json.load(file)

        self.id = meta['id']
        self.title = meta['title']
        self.author = meta['author']
        self.lang = meta['lang']
        self.issue = meta['issue']
        self.vol = meta['vol']
        self.date = meta['date']
        self.type = meta['type']
        self.text = meta['text']

        if 'cleanText' in meta:
            self.cleanText = meta['cleanText']
        
        if 'bib' in meta:
            self.bib = meta['bib']

        if 'bagOfWords' in meta:
            self.bagOfWords = meta['bagOfWords']

    def saveDict(self, path):
        """
        Method to save a JSON version of the class for later use.
        """
        # We save a dictionary version of the Article class.
        with open(f'{path}/{self.id}.json', 'w') as f:
            json.dump(self.__dict__, f)

    # Representación en string de cada artículo, por si acaso.
    def __str__(self):
        return "{} ({}). {}. {}({})".format(self.author, self.date, self.title, self.vol, self.issue)
