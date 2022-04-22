"""
This module contains the exploration utilities
for analyzing the topics.
"""
import json
from corpus import Corpus

# Importing global things

## All articles:
corpus = Corpus()
corpusList = corpus.get_documents_list()

# MODELPATH = "../notebooks/LDA_gensim_90_final.model"
MODELPATH = "../notebooks/models/70_topics/LDA_gensim_70.model"


def prepare_bag_of_words(article):
    """
    A hot fix on some empty strings.
    """
    bow = article.bagOfWords
    bow = bow.split(" ")
    return [w for w in bow if len(w) > 1]


def topics_in_article(lda, article):
    bow = lda.id2word.doc2bow(prepare_bag_of_words(article))
    return lda.get_document_topics(bow)


## Articles in topics:
print("Creating ../data/articles_in_topics.json")
# The model
from gensim.models.ldamodel import LdaModel

lda = LdaModel.load(MODELPATH)

all_topics_in_article = {a.id: topics_in_article(lda, a) for a in corpusList}

articles_in_topics = {topic_id: [] for topic_id in range(90)}
for art_id, topics_and_probs in all_topics_in_article.items():
    for topic_id, prob in topics_and_probs:
        if prob >= 0.01:
            articles_in_topics[topic_id].append((art_id, float(prob)))

# Sorting them by probability
sorted_articles_in_topics = {}
for topic_id, articles_and_probs in articles_in_topics.items():
    sorted_articles_in_topics[topic_id] = sorted(
        articles_and_probs, key=lambda x: x[1], reverse=True
    )

with open("../data/articles_in_topics.json", "w") as fp:
    json.dump(sorted_articles_in_topics, fp)

a_in_topics = {str(k): v for k, v in sorted_articles_in_topics.items()}


def get_articles_in_topic(topic_id, min_prob=0.1, n=None):
    """
    This function returns a list of articles that belong
    with at least {min_prob} to a given topic. If {n}
    is an integer, it will return only the {n} articles.
    """
    a_by_id = {a.id: a for a in corpusList}

    # Filters by probability.
    a_in_topic = a_in_topics[str(topic_id)]
    result = []
    for i, (a_id, prob) in enumerate(a_in_topic):
        if prob >= min_prob:
            result.append((a_by_id[a_id], prob))
        else:
            # because the list is sorted.
            break

        if isinstance(n, int):
            if i == n - 1:
                break

    return result


def get_titles_in_topic(topic_id, min_prob=0.1, n=None):
    articles = get_articles_in_topic(topic_id, min_prob=min_prob, n=n)
    titles = []
    for a, _ in articles:
        if hasattr(a, "title"):
            titles.append(a.title)
        else:
            titles.append("NO TITLE FOUND")

    return titles


def get_keywords_in_topic(topic_id, min_prob=0.1, n=None):
    articles = get_articles_in_topic(topic_id, min_prob=min_prob, n=n)
    keywords = []
    for a, _ in articles:
        if hasattr(a, "keywords"):
            keywords.append(a.keywords)
        else:
            keywords.append("NO KEYWORDS FOUND")

    return keywords


def summary(lda, article, probability=None, topics=False):
    # Print the title
    print("-" * 50)
    print("\t\t TITLE \t\t")
    print(article.title)
    # Print the abstract

    if probability:
        print(f"(with prob. {probability:1.4f})")

    # Print keywords
    print("\t\t KEYWORDS \t\t")
    if hasattr(article, "keywords"):
        print(article.keywords)
    else:
        print("No keywords stored")
    print()

    print("\t\t ABSTRACT \t\t")
    if hasattr(article, "abstract"):
        print(article.abstract)
    else:
        print("No abstract stored")
    print()

    # Print the topics
    if topics:
        print("\t\t TOPICS \t\t")
        topics_in_art = topics_in_article(lda, article)
        for top_id, prob in topics_in_art:
            print(f"Topic {top_id} (w. probability {prob:0.3f})")


def summarize_topic(lda, topic_id, min_prob=0.1, n=None):
    articles = get_articles_in_topic(topic_id, min_prob=min_prob, n=n)
    for a, _ in articles:
        summary(lda, a)


def topic_top_n(lda, topic_id, n=10, verbose=False):
    """
    This function returns a list with
    the top {n} words in a topic given
    a certain lda fit.

    If verbose, it will also print the
    topic using gensim's LDA pretty printer.
    """
    if verbose:
        print(lda.print_topic(topic_id, topn=n))

    return [
        (lda.id2word.get(idx), f"{prob:0.3}")
        for idx, prob in lda.get_topic_terms(topic_id, topn=n)
    ]
