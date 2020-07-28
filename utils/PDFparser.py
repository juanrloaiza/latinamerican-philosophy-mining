#!/usr/bin/env python
# coding: utf-8

import glob
import re
import json
import time
from PyPDF4.pdf import PdfFileReader
from multiprocessing import Pool
import pandas as pd

from nltk.corpus import stopwords
stopwords_es = stopwords.words('spanish') # We load Spanish stopwords.

# We will record data to analyze later.
data = []

from spellchecker import SpellChecker
spell = SpellChecker(language=None)  # We will load a custom dictionary from RAE.and HTML files.

# Note: This version actually includes some manual edits.
# TODO: Document how these manual edits were produced.
spell.word_frequency.load_dictionary('../notebooks/wordlists/customDictionary.gz')

# We add dictionaries in English and German
spell_en = SpellChecker(language = 'en')
spell_de = SpellChecker(language = 'de')

# And we add a dictionary for manual replacements to do a quicker job.
with open('../notebooks/wordlists/pdfManualCorrections.json') as fp:
    manualCorrections = json.load(fp)

def getCorrectedWord(word):
    """
    Function: Takes a word (str) and returns a correction
    using PySpellChecker.
    """

    # If word is recognized in English, return the same word.
    # Often these words have í instead of i, so we also try replacing these.
    if word.lower() in spell_en or word.lower() in spell_de or word.replace('í', 'i').lower() in spell_en:
        return word, word

    # There are patterns where spaces went missing but some are
    # easy to detect. Let's try with "filosofía"
    if len(re.findall('\w*filosof[ií]a\w*', word)) > 0:
        correctedWord = re.sub('(\w*)filosof[íi]a(\w*)', r'\1 filosofía \2', word)
        print(correctedWord)
        return word, correctedWord

    # We will keep track of how long it takes to replace each word.
    # If it takes more than 10 seconds, we will add it to a list of manual
    # replacements that we can do quicker before we pass words to this function.
    #initialTime = time.time()
    correctedWord = spell.correction(word)
    #finalTime = time.time() - initialTime

    #if finalTime > 10:
        #print(f"'{word}' (Sugerencia: {correctedWord})")

    return word, correctedWord

def processText(doc):
    """
    Function: Takes in a document (dict), corrects it, and exports it.
    This function also calculates some useful data and saves it for later analysis.

    Return: correctedText (str)
    """

    correctedText = doc['text']
    doc_id = doc['id']

    # Let's get all non-stopword words.
    totalWords = set([word for word in re.findall('\w+', correctedText) if word not in stopwords_es])

    # We take an initial measure of how many words we recognize over the total count.
    initial = len(spell.unknown(totalWords)) / len(totalWords)

    startTime = time.time()    # We also take an initial timestamp.

    # FIRST CORRECTION
    # We eliminate new lines.
    correctedText = correctedText.replace('\n', ' ')


    # SECOND CORRECTION
    # We replace words that we already know how to replace and that take too
    # long with pyspellshecker.
    for word, replacement in manualCorrections.items():
        correctedText = correctedText.replace(word, replacement)


    # THIRD CORRECTION
    # There are common artifacts derived from OCR. We can eliminate those manually.

    # e.g. A very common pattern is that upper case M is OCR'd as ivi. We can replace
    # with regex. Similar patterns are 1 and l instead of i.
    correctedText = re.sub(r'([A-Z]+)[li1]v[li1]([A-Z]+)', r'\1M\1', correctedText)

    correctedText = re.sub(r'(\w+)[^i]ci[oó]\b', r'\1ción', correctedText)
    correctedText = re.sub(r'\bf[l1Ii](\w+)f[l1Ii]\b', r'\1', correctedText)


    # FOURTH CORRECTION
    # We eliminate consonants that are alone and should not be.
    correctedText = re.sub(r'\b[qwrtpsdfgjklñzxcvbnmQWRTPSDFGJKLÑZXCVBNM]\b', '', correctedText)

    # FIFTH CORRECTION
    # Use PySpellChecker to get corrected words.

    # We take all of the (unique) words in the text.
    words = set(re.findall('\w+', correctedText))

    for word in words:
        if len(word) > 20:
            print(word)

    # We must keep the original word as the spell.unknown function passes everything
    # to lower case, which precludes some replacements later on.
    unknownWords = [word for word in words if word.lower() in spell.unknown(words)]

    # We will consider only unknown words and pass this to the
    # function getting corrected words. We do this in several threads.
    with Pool(6) as pool:
        correctedWords = pool.map(getCorrectedWord, unknownWords)

    # We receive a list of corrections and we implement them.
    for word, correctedWord in correctedWords:
        correctedText = re.sub(r'\b' + word + r'\b', correctedWord, correctedText)


    # FINISHING STEPS
    # We take the word count again since the text has been modified.
    # Again, we don't count stopwords.
    totalWords = set([word for word in re.findall('\w+', correctedText) if word not in stopwords_es])

    # We take a final measure of how much of the text we recognize and how long it took.
    correction = len(spell.unknown(totalWords)) / len(totalWords)
    timeDelta = time.time() - startTime

    # We print out the details.
    print(f"Text ID: {doc_id}")
    print(f"Initial ratio: {initial}")
    print(f"Corrected ratio: {correction}")
    print(f"Time elapsed: {timeDelta:.2f}")

    # We record our results into the data.
    data.append({'id': doc_id,
           'initial': initial,
           'final': correction,
           'time': timeDelta,
            'wordCount': len(re.findall('\w+', correctedText))})

    return correctedText


# MAIN FUNCTION
files = glob.glob('../data/rawPDF/*/*.json')
counter = 0
for file in files:
    with open(file) as fp:
        meta = json.load(fp)

    doc = {}

    doc['id'] = meta['DC.Identifier']
    doc['title'] = meta['description']
    doc['author'] = meta['DC.Creator.PersonalName']
    doc['lang'] = meta['DC.Language']
    doc['issue'] = meta['DC.Source.Issue']
    doc['vol'] = meta['DC.Source.Volume']
    doc['date'] = meta['citation_date']
    doc['type'] = meta['DC.Type.articleType']

    # We create some attributes to fit the class into the Article class model.
    # We initialize them as None though.
    doc['cleanText'] = None
    doc['bagOfWords'] = None

    if 'keywords' in meta.keys():
        doc['keywords'] = meta['keywords']

    if doc['lang'] != 'es':
        continue

    pdfPath = file.replace('json', 'pdf')
    print(pdfPath)
    reader = PdfFileReader(open(pdfPath, 'rb'))

    text = ''
    for page in range(reader.getNumPages()):
        text += reader.getPage(page).extractText()

    doc['text'] = text

    # Some text only has whitespace characters. It is length > 0 but no meaningful
    # characters. We eliminate those by checking if the text splits into something.
    # If it splits, we process it. Otherwise, we save None as the text value.
    if text.split():
        doc['text'] = processText(doc)
    else:
        doc['text'] = None

    # We save the final version of the document as JSON.
    with open(f'../data/parsedPDF/{doc_id}.json', 'w') as fp:
        json.dump(doc, fp)

    print(f'Saved {doc_id}.json')

    counter += 1
    print(f"{counter} of {len(files)} revised.")
    print('---------------------')

# We save collected data for analysis as csv.
df = pd.DataFrame(data)
df.to_csv('../data/PDFCorrections.csv')
