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

        # Gensim can filter the dictionary to remove rare tokens. This helps reduce the computation.
        self.id2word.filter_extremes(no_below=5, no_above=0.99)

        self.lda = None
        self.ldaseq = None

    def train(self, seed=None, workers=5, dtm=True, time_window: int = 5):
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
            eval_every=None,
        )

        print("Static model trained!")

        if dtm:
            self.ldaseq = DtmModel(
                dtm_path="utils/dtm-linux64",
                corpus=corpus_bows,
                time_slices=self.corpus_obj.get_time_slices(time_window=time_window),
                num_topics=self.num_topics,
                id2word=self.id2word,
                rng_seed=seed,
            )
            print("Sequential model trained!")

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

        print("Creating Topic objects...")
        self.create_topics()

    def load(self):
        """Loads a model from a .model file.

        TODO: Right now we load one model per num. of topics. Maybe we can load from a path instead?

        TODO: Loading sequential model.
        """
        self.lda = LdaMulticore.load(
            f"gensim_models/gensim_{self.num_topics}/LDA_gensim_{self.num_topics}.model"
        )

        self.ldaseq = DtmModel.load(
            f"gensim_models/gensim_{self.num_topics}/LDA_gensim_{self.num_topics}.dmodel"
        )
        self.id2word = self.lda.id2word
        self.create_topics()

    def save(self):
        """Saves the model to a .model file, including the dictionary."""
        folder = f"gensim_models/gensim_{self.num_topics}"
        path = f"{folder}/LDA_gensim_{self.num_topics}"

        if not os.path.exists(folder):
            os.makedirs(folder)

        if isinstance(self.lda, LdaMulticore):
            self.lda.save(f"{path}.model")

        if isinstance(self.ldaseq, DtmModel):
            self.ldaseq.save(f"{path}.dmodel")

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

    def print_topics(self, probability_mass=None):
        for id, topic in self.lda.show_topics(
            num_words=1000, log=False, num_topics=-1, formatted=False
        ):
            print(f"Topic #{id}")
            print("----------")
            if probability_mass:
                printed_mass = 0
                for word, prob in topic:
                    print(f"{word}: {prob:.5f}")
                    printed_mass += prob
                    if printed_mass > probability_mass:
                        break
            else:
                for word, prob in topic:
                    print(f"{word}: {prob:.5f}")
            print("\n")

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

    def get_difference_matrix(self, num_words=20):
        """Returns the difference matrix for the model using Jaccard distance (1 - Jaccard similarity)."""
        return self.lda.diff(self.lda, num_words=num_words, distance="jaccard")[0]

    def get_topic_masses(self, mass=0.5):
        """Returns the mean amount of words that accounts for each topic's probability mass."""
        return [len(topic.get_top_words(mass=mass)) for topic in self.topics]

    def get_stats(self):
        """Returns statistics regarding the model. Useful for analysis when calibrating."""

        # Train the model and time it.
        print("Timing training...")
        start_train = time.time()
        self.train(dtm=False)
        end_train = time.time()

        # Get the model C_V coherence
        print("Getting model coherence...")
        start_coherence = time.time()
        coherence = self.get_coherence()
        end_coherence = time.time()

        # Get how many articles it has in each topic for a mean and how many empty topics there are.
        arts_per_topic = [len(topic) for topic in self.topics]
        empty_topics = len([n for n in arts_per_topic if n == 0])

        # Get metrics for the difference matrix of the model.
        diff_norm_score = np.linalg.norm(self.get_difference_matrix()) / self.num_topics
        diff_eig_score = (
            np.linalg.eig(self.get_difference_matrix())[0][0] / self.num_topics
        )

        return {
            "coherence": coherence,
            "log_perplexity": self.get_log_perplexity(),
            "time_lda": end_train - start_train,
            "time_coherence": end_coherence - start_coherence,
            "orphans": len(self.get_orphans()),
            "empty_topics": empty_topics,
            "avg_arts_per_topic": np.mean(arts_per_topic),
            "std_arts_per_topic": np.std(arts_per_topic),
            "min_arts_in_topic": np.min(arts_per_topic),
            "max_arts_in_topic": np.max(arts_per_topic),
            "diff_norm_score": diff_norm_score,
            "diff_eig_score": diff_eig_score,
            "topic_mass_length": np.mean(self.get_topic_masses()),
        }


if __name__ == "__main__":
    corpus = Corpus(registry_path="../utils/article_registry.json")
    N_TOPICS = 10
    base_model = Model(corpus, N_TOPICS)
