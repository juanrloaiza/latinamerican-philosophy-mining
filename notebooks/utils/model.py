import time
import subprocess
from pathlib import Path
import numpy as np
from tqdm import tqdm
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
from utils.topic import Topic
from utils.corpus import Corpus

models_path = Path(__file__).parent.parent.resolve() / "models"
models_path.mkdir(exist_ok=True)


class Model:
    def __init__(
        self, corpus: Corpus, num_topics: int, time_window: int = 10, seed: int = None
    ) -> None:
        # Generate a model id and seed
        if seed:
            self.seed = seed
        else:
            self.seed = np.random.randint(1, 99999)

        self.path = models_path / f"{num_topics}_{self.seed}"

        # Load the corpus and get corpus properties.
        self.corpus = corpus
        self.num_docs = len(corpus)
        self.num_topics = num_topics

        # Get number of docs per time slice and total number of slices for a given time window.
        self.slices = corpus.get_time_slices(time_window)
        self.num_slices = len(self.slices)

        # Build a dictionary
        self.id2word = corpora.Dictionary(
            [doc.get_bow_list() for doc in corpus.documents]
        )

        # Gensim can filter the dictionary to remove rare tokens. This helps reduce the computation.
        self.id2word.filter_extremes(no_below=5, no_above=0.99)

        # Initialize topic list
        self.topics = []

    def prepare_corpus(self) -> None:
        """Prepares two files that the model needs for training:

        - corpus-mult.dat   A "len {word:count}" representation of each document in the corpus.
            Example: 293 1:20 2:50 3:0 4:912...

        - corpus-seq.dat    A list of time slices and number of documents per time slice
                            (sum = total docs in corpus)
            Example: 291 252 305...
        """

        # Get {word: count} representation of each document
        output_str = ""
        for doc in self.corpus.documents:
            doc_bow = doc.get_bow_list()
            word_counts = [
                f"{idx}:{count}" for idx, count in self.id2word.doc2bow(doc_bow)
            ]
            output_str += (
                f"{len(self.id2word.doc2bow(doc_bow))} " + " ".join(word_counts) + " \n"
            )

        with open(models_path / "corpus-mult.dat", "w") as file:
            file.write(output_str)

        # Get number of documents per time slice
        with open(models_path / "corpus-seq.dat", "w") as file:
            file.write(f"{self.num_slices}\n" + "\n".join(map(str, self.slices)))

    def train(self) -> None:
        """Trains the model.

        Runs Blei-lab's binary and produces outputs in file."""

        self.path.mkdir(exist_ok=True)

        kwargs = {
            "model": "dtm",
            "ntopics": self.num_topics,
            "mode": "fit",
            "rng_seed": self.seed,
            "initialize_lda": "true",
            "corpus_prefix": str(models_path / "corpus"),
            "outname": str(self.path),
            "top_chain_var": 0.005,
            "alpha": 0.01,
            "lda_sequence_min_iter": 6,
            "lda_sequence_max_iter": 20,
            "lda_max_em_iter": 10,
        }

        training_process = subprocess.run(
            ["dtm_files/dtm-linux64"] + [f"--{kw}={val}" for kw, val in kwargs.items()],
            check=True,
        )

        print("Model trained! Loading topics...")
        self.load_topics()

    def load_topics(self) -> None:
        """Creates Topic objects for each topic in the model. This allows us to
        interface with the topics directly through methods defined in the Topic class."""
        self.topics = [
            Topic(topic_num, self.num_slices, self.id2word, model_path=self.path)
            for topic_num in tqdm(range(self.num_topics))
        ]
        self.classify_documents()

    def classify_documents(self) -> None:
        """Classifies documents into the topics in the model."""
        gamma = np.loadtxt(self.path / "lda-seq" / "gam.dat").reshape(self.num_docs, -1)
        gamma = gamma / gamma.sum(axis=1, keepdims=True)

        normalized_gamma = gamma / gamma.sum(axis=0)

        # Classify using likelihood sorting

        for topic in self.topics:
            topic.docs = []
            indices = gamma[:, topic.topic_id].argsort()[::-1]

            checked_mass = 0
            for idx in indices:
                topic.docs.append(self.corpus.documents[idx])
                checked_mass += normalized_gamma[idx, topic.topic_id]

                if checked_mass > 0.5:
                    break

    def get_coherence(self) -> list:
        """Computes the coherence of the model per topic."""
        # TODO: Check if we built the Topic objects already before this loop.

        top_words_per_slice = {
            time_slice: [
                topic.top_word_evolution_table(20)[time_slice].to_list()
                for topic in self.topics
            ]
            for time_slice in range(self.num_slices)
        }

        texts = [doc.get_bow_list() for doc in self.corpus.documents]

        coherences_per_slice = {}
        for time_slice, top_words in top_words_per_slice.items():
            coherence_model = CoherenceModel(
                topics=top_words,
                texts=texts,
                dictionary=self.id2word,
                coherence="c_v",
            )

            coherences_per_slice[time_slice] = coherence_model.get_coherence_per_topic()

        return coherences_per_slice

    def get_orphans(self) -> set:
        """Returns a list of articles without a topic."""
        assigned_docs = set()
        orphans = set()
        for doc in self.corpus.documents:
            for topic in self.topics:
                if doc.id in [doc.id for doc in topic.docs]:
                    assigned_docs.add(doc.id)
                    break

            if doc.id not in assigned_docs:
                orphans.add(doc.id)

        return orphans

    def get_doc_masses(self, mass: int = 0.5) -> list:
        """Returns the amount of documents that account for `mass` probability mass for all topics."""
        return [topic.get_doc_mass(mass) for topic in self.topics]

    def get_word_masses(self, mass: int = 0.5) -> float:
        """Returns the amount of words that account for `mass` probability mass of a topic."""
        return [topic.get_word_mass(mass) for topic in self.topics]

    def get_stats(self) -> dict:
        """Returns statistics regarding the model. Useful for analysis when calibrating."""

        # Train the model and time it.
        print("Timing training...")
        start_train = time.time()
        self.train()
        end_train = time.time()

        # Get the model C_V coherence
        print("Getting model coherence...")
        start_coherence = time.time()
        coherence = self.get_coherence()
        end_coherence = time.time()

        # Get how many articles it has in each topic for a mean and how many empty topics there are.
        arts_per_topic = [len(topic.docs) for topic in self.topics]
        empty_topics = len([n for n in arts_per_topic if n == 0])

        return {
            "coherence": coherence,
            "time_lda": end_train - start_train,
            "time_coherence": end_coherence - start_coherence,
            "orphans": len(self.get_orphans()),
            "empty_topics": empty_topics,
            "avg_arts_per_topic": np.mean(arts_per_topic),
            "std_arts_per_topic": np.std(arts_per_topic),
            "min_arts_in_topic": int(np.min(arts_per_topic)),
            "max_arts_in_topic": int(np.max(arts_per_topic)),
            "avg_doc_mass": np.mean(self.get_doc_masses()),
            "avg_word_mass": np.mean(self.get_word_masses()),
        }

    def export_summary(self, filename: str):
        """TODO: Saves a summary file in PDF format for later usage."""
        pass


if __name__ == "__main__":
    N_TOPICS = 10
    base_model = Model(Corpus(registry_path="../utils/article_registry.json"), N_TOPICS)
