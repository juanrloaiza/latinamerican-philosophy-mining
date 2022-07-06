import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


class Topic:
    """An interface for topics in a given model."""

    def __init__(
        self, topic_id: int, num_slices: int, id2word: dict[int, str], model_path: Path
    ) -> None:
        self.topic_id = topic_id
        self.num_slices = num_slices
        self.model_path = model_path

        # Initialize data attributes
        self.word_table = None
        self.full_data = None
        self.docs = []

        # Load the word distributions from file
        # TODO: Improve loading paths and file naming.
        self.word_probabilities = np.loadtxt(
            self.model_path
            / "lda-seq"
            / f"topic-{self.topic_id:03d}-var-e-log-prob.dat"
        ).reshape(-1, num_slices)

        # Distributions are in log prob form. Convert to [0, 1] probs.
        self.word_probabilities = np.exp(self.word_probabilities)

        # Sort by mean probability across all time slices
        sorted_idxs = self.word_probabilities.mean(axis=1).argsort()[::-1]

        # Add words until we hit half of the topic's probability mass (0.5/1.0)
        # or there are 2000 words in the topic (which would lead to eliminating it).
        checked_probability = 0
        data = []
        word_table_raw = {}
        for idx in sorted_idxs:

            if checked_probability >= 0.5 or len(word_table_raw) > 2000:
                break

            word = id2word[idx]
            word_table_raw[word] = []
            for time_slice, prob in enumerate(self.word_probabilities[idx]):
                data.append(
                    {
                        "Word": word,
                        "Slice": time_slice,
                        "Prob": prob,
                        "Pos": np.where(
                            self.word_probabilities[:, time_slice].argsort()[::-1]
                            == idx
                        )[0][0]
                        + 1,
                    }
                )

                word_table_raw[word].append(prob)

            checked_probability += np.mean(self.word_probabilities[idx])

        self.full_data = pd.DataFrame(data)

        self.word_table = pd.DataFrame()
        for word, probs in word_table_raw.items():
            word_probs = pd.Series(probs, name=word)
            self.word_table = pd.concat([self.word_table, word_probs], axis=1)

        self.word_table = self.word_table.T
        self.word_table.insert(0, "Mean prob.", self.word_table.mean(axis=1))
        self.length = len(self.word_table)

    def top_words(self, n=10) -> pd.Series:
        """Returns the top n words for the whole topic (averaged across time slices)."""
        return (
            self.full_data.groupby("Word")["Prob"]
            .mean()
            .sort_values(ascending=False)[:n]
        )

    def get_doc_mass(self, mass: int = 0.5) -> float:
        """Returns the amount of documents that account for `mass` probability mass of a topic."""

        # TODO: Use mass parameter

        return len(self.docs)

    def get_word_mass(self, mass: int = 0.5) -> float:
        """Returns the amount of words that account for `mass` probability mass of a topic."""
        return self.word_table.shape[0]

    def top_words_evolution_graph(self) -> plt.Axes:
        """Returns a graph of the evolution of top words per time slice."""
        df = self.full_data

        ax = sns.pointplot(
            data=df[df["Pos"] <= 10], x="Slice", y="Pos", hue="Word", palette="tab20"
        )
        plt.legend(bbox_to_anchor=(1, 1))
        plt.ylim(reversed(plt.ylim()))
        sns.despine()

        return ax

    def top_word_evolution_table(self, n=10) -> pd.DataFrame:
        """Returns a DataFrame of top words for each time slice."""
        slice_lists = pd.DataFrame()
        for time_slice in range(self.num_slices):
            df = self.full_data
            slice_lists[time_slice] = list(
                df[df["Slice"] == time_slice].sort_values("Pos")["Word"][:n]
            )

        return slice_lists
