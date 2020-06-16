#!/usr/bin/env python
# coding: utf-8

"""
This file includes utility functions to load the corpus in various ways in
which we will analyze it. We can import these functions in different scripts.
"""

import os
import json

# Default path for files: ../data/clean_json/

def loadCorpusList(path):
    """
    This function takes in a path and loads every file from clean JSON files.
    It returns a list with dictionaries for each entry along with each article's
    metadata.

    Input: Path (string)
    Output: Dictionaries for each article (list)
    """

    corpusList = []
    for file in os.listdir(path):
        with open(f"{path}/{file}", 'r') as fp:
            corpusList.append(json.load(fp))

    return corpusList


def loadCorpusDict(path):
    """
    This function takes in a path and loads every file from clean JSON files.
    It returns a dictionary where the keys are article IDs and values are article
    texts.

    Input: Path (string)
    Output: Article ID and text (dictionary)
    """

    corpusDict = {}
    for file in os.listdir(path):
        with open(f"{path}/{file}", 'r') as fp:
            article = json.load(fp)
            corpusDict[article['id']] = article['text']

    return corpusDict
