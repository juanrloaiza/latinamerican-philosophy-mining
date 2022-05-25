import os
import pandoc
import time
import numpy as np
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.ldaseqmodel import LdaSeqModel
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
from utils.topic import Topic
from utils.corpus import Corpus
from utils.dtmmodel import DtmModel


class Model:
    def __init__(self, corpus: Corpus, num_topics: int):
        self.corpus_obj = corpus
        self.num_topics = num_topics
        self.topics = []  # Topic objects
        self.articles = self.corpus_obj.get_documents_list()
        self.bows = {article.id: article.get_bow_list() for article in self.articles}
        self.id2word = corpora.Dictionary(self.bows.values())

    def train(self, seed=None, workers=5, time_window: int = 5):
        """Trains the model."""
        corpus_bows = [self.id2word.doc2bow(text) for text in self.bows.values()]

        print("Bags of words collected. Starting training...")

        self.lda = LdaMulticore(
            corpus_bows,
            num_topics=self.num_topics,
            id2word=self.id2word,
            passes=15,
            random_state=seed,
            workers=workers,
        )

        print("Base model trained. Training sequential model....")

        """
        self.ldaseq = LdaSeqModel(
            corpus=corpus_bows,
            time_slice=self.corpus_obj.get_time_slices(time_window=time_window),
            num_topics=self.num_topics,
            id2word=self.id2word,
            passes=15,
            initialize="ldamodel",
            lda_model=self.lda,
            random_state=seed,
        )"""

        self.ldaseq = DtmModel(
            dtm_path="utils/dtm-linux64",
            corpus=corpus_bows,
            time_slices=self.corpus_obj.get_time_slices(time_window=time_window),
            num_topics=self.num_topics,
            id2word=self.id2word,
            rng_seed=0,
        )

        print("Sequential model trained! Creating Topic objects...")
        self.create_topics()

    def load(self):
        """Loads a model from a .model file.

        TODO: Right now we load one model per num. of topics. Maybe we can load from a path instead?
        """
        self.lda = LdaMulticore.load(
            f"gensim_models/gensim_{self.num_topics}/yLDA_gensim_{self.num_topics}.model"
        )
        self.id2word = self.lda.id2word
        self.create_topics()

    def save(self):
        """Saves the model to a .model file, including the dictionary."""
        folder = f"gensim_models/gensim_{self.num_topics}"
        path = f"{folder}/yLDA_gensim_{self.num_topics}"

        if not os.path.exists(folder):
            os.makedirs(folder)

        self.lda.save(f"{path}.model")
        self.id2word.save(f"{path}.txt")
        print(f"Saved to: {path}")

    def get_topics_in_article(self, bow, min_prob=0.01):
        """Returns all the topic id, probability tuples given an Article object."""
        bow_matrix = self.id2word.doc2bow(bow)
        return self.lda.get_document_topics(
            bow_matrix,
            minimum_probability=min_prob,
        )

    def create_topics(self):
        """Creates Topic objects for each topic in the model. This allows us to
        interface with the topics directly through methods defined in the Topic class."""
        topics_dict = {topic_id: [] for topic_id in range(self.num_topics)}

        topics_in_articles = {
            id: self.get_topics_in_article(text) for id, text in self.bows.items()
        }

        for art_id, topics_and_probs in topics_in_articles.items():
            for topic_id, prob in topics_and_probs:
                topics_dict[topic_id].append((art_id, float(prob)))

        for topic_id, article_list in topics_dict.items():
            new_topic = Topic(
                topic_id,
                sorted(article_list, key=lambda x: x[1], reverse=True),
                model=self,
            )
            self.topics.append(new_topic)

    def get_article_ref(self, article_id):
        """Gets the article reference from the Corpus."""
        return self.corpus_obj.get_article_ref(article_id)

    def export_summary(self, filename: str):
        """Saves a summary file in PDF format for later usage."""
        summary = ""
        for topic in self.topics:
            summary += topic.summary() + "\n\\newpage"
        doc = pandoc.read(summary)
        pandoc.write(doc, file=filename, format="pdf")

    def get_coherence(self):
        """Computes the coherence of the model."""
        coherence_model = CoherenceModel(
            model=self.lda,
            texts=self.bows.values(),
            dictionary=self.id2word,
            coherence="c_v",
        )
        return coherence_model.get_coherence()

    def get_log_perplexity(self):
        """Returns the log perplexity of the model."""
        corpus_bows = [self.id2word.doc2bow(text) for text in self.bows.values()]
        return self.lda.log_perplexity(corpus_bows)

    def get_orphans(self):
        """Returns a list of articles without a topic."""
        return [
            article
            for article in self.articles
            if not self.get_topics_in_article(self.bows[article.id])
        ]

    def get_stats(self):
        """Returns statistics regarding the model. Useful for analysis when calibrating."""
        print("Timing training...")
        start_train = time.time()
        self.train()
        end_train = time.time()

        print("Getting model coherence...")
        start_coherence = time.time()
        c = self.get_coherence()
        end_coherence = time.time()

        arts_per_topic = [len(topic) for topic in self.topics]

        return {
            "coherence": c,
            "log_perplexity": self.get_log_perplexity(),
            "time_lda": end_train - start_train,
            "time_coherence": end_coherence - start_coherence,
            "orphans": len(self.get_orphans()),
            "avg_arts_per_topic": np.mean(arts_per_topic),
            "std_arts_per_topic": np.std(arts_per_topic),
            "min_arts_in_topic": np.min(arts_per_topic),
            "max_arts_in_topic": np.max(arts_per_topic),
        }


if __name__ == "__main__":
    corpus = Corpus(registry_path="../utils/article_registry.json")
    n_topics = 10
    base_model = Model(corpus, n_topics)
