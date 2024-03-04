from typing import Dict, List
import multiprocessing
import time
import subprocess
from pathlib import Path
import platform
import json
import pickle
from collections import defaultdict

import numpy as np

from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora

from utils.topic import Topic
from utils.corpus import Corpus


models_path = Path(__file__).parent.parent.resolve() / "models"
models_path.mkdir(exist_ok=True)

NOTEBOOKS_DIR = Path(__file__).parent.parent.resolve()


class Model:
    def __init__(
        self,
        corpus: Corpus,
        num_topics: int,
        time_window: int = 10,
        seed: int = None,
        custom_output_path: Path = None,
    ) -> None:
        # Generate a model id and seed
        if seed:
            self.seed = seed
        else:
            self.seed = np.random.randint(1, 99999)

        if custom_output_path:
            self.path = custom_output_path
        else:
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

    def prepare_corpus(self, force: bool = False) -> None:
        """Prepares two files that the model needs for training:

        - corpus-mult.dat   A "len {word:count}" representation of each document in the corpus.
            Example: 293 1:20 2:50 3:0 4:912...

        - corpus-seq.dat    A list of time slices and number of documents per time slice
                            (sum = total docs in corpus)
            Example: 291 252 305...
        """
        if (
            (models_path / "corpus-mult.dat").exists()
            and (models_path / "corpus-seq.dat").exists()
            and not force
        ):
            return

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

        if platform.system() == "Darwin":
            print("Running on MacOSX. Using binary dtm-darwin64.")
            binary_name = "dtm-darwin64"
        else:
            print("Defaulting to Linux. Using binary dtm-linux64.")
            binary_name = "dtm-linux64"

        print(
            f"Command to run: "
            f"{NOTEBOOKS_DIR}/dtm_files/{binary_name} "
            + " ".join([f"--{kw}={val}" for kw, val in kwargs.items()])
        )

        try:
            training_process = subprocess.run(
                [f"{NOTEBOOKS_DIR}/dtm_files/{binary_name}"]
                + [f"--{kw}={val}" for kw, val in kwargs.items()],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            print(e.stderr.decode("utf-8"))
            raise e

        print("Model trained! Loading topics...")
        self.load_topics()

    def load_topics(self, num_workers: int = 5) -> None:
        """Creates Topic objects for each topic in the model. This allows us to
        interface with the topics directly through methods defined in the Topic class.
        """

        # This process is taking a whole minute.
        # TODO: Check for ways to optimize it besides multithreading.
        CACHE_FOR_TOPICS = NOTEBOOKS_DIR.parent.resolve() / "data" / "topics_cache"
        CACHE_FOR_TOPICS.mkdir(exist_ok=True)
        PATH_FOR_CACHE = CACHE_FOR_TOPICS / f"topics_{self.num_topics}_{self.seed}.pkl"
        if PATH_FOR_CACHE.exists():
            print("Loading topics from cache")
            with open(PATH_FOR_CACHE, "rb") as fp:
                topics = pickle.load(fp)
        else:
            if num_workers > 1:
                pool = multiprocessing.Pool(num_workers)
                topics = pool.map(self.create_topic, range(self.num_topics))
            else:
                topics = [self.create_topic(i) for i in range(self.num_topics)]

            # Saving cache
            with open(PATH_FOR_CACHE, "wb") as fp:
                print("Saving topics from cache")
                pickle.dump(topics, fp)

        self.topics = topics

        # Reads the results from the DTM binary.
        self.classify_documents()

        # Attaches the tags to the topics.
        self.tag_topics()

    def create_topic(self, topic_num: int) -> Topic:
        "Creates a Topic object from a topic number."
        return Topic(
            topic_num,
            self.num_slices,
            self.id2word,
            model_path=self.path,
            time_slice_years=self.corpus.time_slice_years,
        )

    def get_main_areas(self) -> Dict[str, List[Topic]]:
        """Returns a dictionary with the main areas and the topics that belong to them."""
        main_areas = defaultdict(list)
        for topic in self.topics:
            if topic.is_trash:
                continue

            main_areas[topic.main_area.capitalize()].append(topic)

        return main_areas

    def count_docs_per_main_area(self, main_area: str) -> Dict[str, int]:
        counts_per_area = defaultdict(int)
        for topic in filter(lambda t: not t.is_trash, self.topics):
            if main_area.lower() == topic.main_area.lower():
                for area in topic.areas:

                    # We exclude special tags like #author or #historical
                    if "#" in area:
                        continue
                    counts_per_area[area.capitalize()] += 1

        return counts_per_area

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
                doc_likelihood = normalized_gamma[idx, topic.topic_id]
                topic.docs.append((self.corpus.documents[idx], doc_likelihood))
                checked_mass += doc_likelihood

                if checked_mass > 0.5:
                    break

            # Mark the topic as garbage if we have more than 200 documents.
            if len(topic.docs) > 200:
                topic.is_trash = True

    def tag_topics(self) -> None:
        """Reads the json file with tags, and attaches them
        to the topics."""
        path_to_json_file = (
            NOTEBOOKS_DIR / "results" / f"topic_tags_{self.num_topics}_{self.seed}.json"
        )

        if not path_to_json_file.exists():
            return

        with open(path_to_json_file) as fp:
            tags_ = json.load(fp)

        for topic_id, tags in tags_.items():
            if len(tags) == 0:
                continue

            topic_id = int(topic_id)

            # Normalize string capitalization before assignment
            tags = [t.capitalize() for t in tags]
            self.topics[topic_id].tags = tags
            self.topics[topic_id].main_area = tags[0]
            self.topics[topic_id].areas = tags[1:]

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
                if doc.id in [doc.id for doc, likelihood in topic.docs]:
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

        # TODO: Document criteria for empty topics
        empty_topics = []
        epsilon = 0.05  # Arbitrary constant
        for topic in self.topics:
            topic_likelihoods = [likelihood for doc, likelihood in topic.docs]
            if max(topic_likelihoods) < ((1 / self.num_docs) + epsilon):
                empty_topics.append(topic)

        return {
            "seed": self.seed,
            "coherence": coherence,
            "time_lda": end_train - start_train,
            "time_coherence": end_coherence - start_coherence,
            "orphans": len(self.get_orphans()),
            "empty_topics": len(empty_topics),
            "avg_arts_per_topic": np.mean(arts_per_topic),
            "std_arts_per_topic": np.std(arts_per_topic),
            "min_arts_in_topic": int(np.min(arts_per_topic)),
            "max_arts_in_topic": int(np.max(arts_per_topic)),
            "avg_doc_mass": np.mean(self.get_doc_masses()),
            "avg_word_mass": np.mean(self.get_word_masses()),
        }

    def summarize(
        self, path_to_save: Path = None, omit_trash_topics: bool = True
    ) -> str:
        """
        Returns a summary of all topics in the model.
        It loops through the topics and gets a summary of each
        one of them with the topic.summarize() method.

        If a filename is provided, the summary is written
        on it as a markdown file.
        """
        non_trash_topic_summaries = [
            topic.summarize() for topic in self.topics if not topic.is_trash
        ]
        trash_topic_summaries = [
            topic.summarize() for topic in self.topics if topic.is_trash
        ]

        summary = (
            f"""
Number of trash topics: {len(trash_topic_summaries)}
            """
            + "\n"
        )
        summary += "\n".join(non_trash_topic_summaries)

        if not omit_trash_topics:
            summary += "\n".join(trash_topic_summaries)

        if path_to_save is not None:
            with open(path_to_save, "w") as fp:
                fp.write(summary)

        return summary


if __name__ == "__main__":
    # N_TOPICS = 10
    # base_model = Model(Corpus(registry_path="../utils/article_registry.json"), N_TOPICS)
    corpus = Corpus(registry_path=NOTEBOOKS_DIR / "utils" / "article_registry.json")

    n_topics = 90
    seed = 36775

    model = Model(corpus, n_topics, seed=seed)
    model.load_topics(num_workers=5)

    model.summarize(NOTEBOOKS_DIR / "results_non_trash.md", omit_trash_topics=True)
