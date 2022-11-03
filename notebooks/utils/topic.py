from typing import List, Tuple

import pandas as pd
import numpy as np
from pathlib import Path


class Topic:
    """An interface for topics in a given model."""

    def __init__(
        self,
        topic_id: int,
        num_slices: int,
        id2word: dict[int, str],
        model_path: Path,
        time_slice_years: List[Tuple[int]] = None,
    ) -> None:
        self.topic_id = topic_id
        self.num_slices = num_slices
        self.model_path = model_path

        # Initialize data attributes
        self.word_table = None
        self.full_data = None
        self.time_slice_years = time_slice_years
        self.docs = []
        self.is_trash = False

        # Load the word distributions from file
        # TODO: Improve loading paths and file naming.
        self.word_probabilities = np.loadtxt(
            self.model_path
            / "lda-seq"
            / f"topic-{self.topic_id:03d}-var-e-log-prob.dat"
        ).reshape(-1, num_slices)

        # Distributions are in log prob form. Convert to [0, 1] probs.
        self.word_probabilities = np.exp(self.word_probabilities)

        sorted_idxs = self.word_probabilities.mean(axis=1).argsort()[::-1]

        mean_probabilities = self.word_probabilities.mean(axis=1)

        mean_probabilities.sort()

        mean_probabilities = mean_probabilities[::-1]

        # Counting all words that account for 0.5 of probablity mass.
        target_idx = min((mean_probabilities.cumsum() >= 0.5).argmax(), 2000) + 1
        self.half_mass_index = target_idx

        # Having at least 10 words in the description.
        target_idx = max(10, target_idx)

        topic_words = [id2word[idx] for idx in sorted_idxs[:target_idx]]

        self.word_table = pd.DataFrame(
            self.word_probabilities[sorted_idxs[:target_idx]],
            index=topic_words,
            columns=time_slice_years,
        )

        self.length = len(self.word_table)

        if self.length > 2000 or self.word_probabilities[0].mean() > 0.75:
            self.is_trash = True

    def top_words(self, n=10) -> np.ndarray:
        """Returns the top n words for the whole topic (averaged across time slices)."""
        return self.word_table[:n].index.values

    def get_doc_mass(self, mass: int = 0.5) -> int:
        """Returns the amount of documents that account for `mass` probability mass of a topic."""

        # TODO: Use mass parameter

        return len(self.docs)

    def get_word_mass(self, mass: int = 0.5) -> int:
        """Returns the amount of words that account for `mass` probability mass of a topic."""
        return self.word_table.shape[0]

    def top_word_evolution_table(self, n=10) -> pd.DataFrame:
        """Returns a DataFrame of top words for each time slice."""
        data = []
        for time_slice in self.word_table:
            data.append(
                self.word_table[time_slice].sort_values(ascending=False)[:n].index
            )

        return pd.DataFrame(data, index=self.time_slice_years).T
