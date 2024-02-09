"""
Implements the utilities around visualizing topics and models.
"""

from typing import Dict
from collections import Counter
import datetime as dt
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import wordcloud as wc
import numpy as np

from utils.topic import Topic
from utils.corpus import Article
from utils.model import Model

FIG_SIZE = (10, 5)
COLORS_FOR_WORD_EVOLUTION = [
    "#8eac65",
    "#f4b393",
    "#140152",
    "#22007c",
    "#0d00a4",
    "#321325",
    "#fcdc4d",
    "#e01a4f",
    "#53b3cb",
    "#a3e7fc",
    "#d6fff6",
    "#231651",
    "#4dccbd",
    "#2374ab",
    "#ff8484",
]


class Visualizer:
    def __init__(self, model: Model) -> None:
        self.model = model

    def count_docs_per_area(self, main_area: str) -> Dict[str, int]:
        counts_per_area = defaultdict(int)
        for topic in filter(lambda t: not t.is_trash, self.model.topics):
            if main_area.lower() == topic.main_area.lower():
                for area in topic.areas:
                    counts_per_area[area.capitalize()] += 1

        return counts_per_area

    @staticmethod
    def plot_number_of_documents_per_year(
        topic: Topic, ax: plt.Axes = None
    ) -> plt.Axes:
        """ """
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=FIG_SIZE)

        counts = topic.count_documents_per_year()
        ax.bar(counts.keys(), counts.values())

    def plot_wordcloud_per_main_area(self, main_area: str, ax: plt.Axes = None) -> None:
        """Generates a word cloud given a main area string."""

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=FIG_SIZE)

        main_area_articles = []
        checked_article_ids = set()
        for topic in self.model.topics:
            if topic.is_trash:
                continue

            if topic.main_area.lower() != main_area.lower():
                continue

            for article, _ in topic.docs:
                if article.id in checked_article_ids:
                    continue
                main_area_articles.append(article)
                checked_article_ids.add(article.id)

        text = " ".join(article.get_bag_of_words() for article in main_area_articles)

        wordcloud = wc.WordCloud().generate_from_text(text)

        ax.imshow(wordcloud)

    def plot_stream_graph(self, ax: plt.Axes = None):
        data = []
        for topic in self.model.topics:
            for doc, _ in topic.docs:
                if topic.tags:
                    data.append(
                        (
                            topic.tags[0].capitalize(),
                            dt.datetime.strptime(doc.date, "%Y/%m/%d").year,
                        )
                    )

        df = pd.DataFrame(data, columns=["Main area", "Date"])
        df.groupby(["Date", "Main area"]).size().unstack().plot(
            kind="area", stacked=True, ax=ax
        )

    def plot_word_evolution_by_topic_graph(self, topic_id: int, ax: plt.Axes = None):

        topic = self.model.topics[topic_id]
        document_count_per_year = topic.count_documents_per_year()

        data = {}
        for time_slice, order in topic.top_word_evolution_table().items():
            # If we don't have any documents in this timeslice, we shouldn't
            # be plotting.
            initial_year, final_year = time_slice
            document_counts = sum(
                [document_count_per_year[y] for y in range(initial_year, final_year)]
            )

            for pos, word in order.items():
                if word not in data:
                    data[word] = {time_slice: np.NaN}

                if document_counts != 0:
                    data[word][time_slice] = pos
                else:
                    data[word][time_slice] = np.NaN

        print(data)
        word_pos_by_timeslice_df = pd.DataFrame.from_dict(data, orient="index").T + 1

        mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=COLORS_FOR_WORD_EVOLUTION)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=FIG_SIZE)
        ax.invert_yaxis()

        # TODO: add time slices as x axis.
        word_pos_by_timeslice_df.plot(ax=ax, marker="o", linestyle="-", markersize=5)
        ax.legend(bbox_to_anchor=(1.1, 1.0))
        ax.set_xlim(-0.5, word_pos_by_timeslice_df.shape[0])
        ax.set_xticks(range(word_pos_by_timeslice_df.shape[0]))
        ax.set_xticklabels(
            [f"{time_slice}" for time_slice in word_pos_by_timeslice_df.index],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )


if __name__ == "__main__":
    from pathlib import Path

    from utils.corpus import Corpus
    from utils.model import Model

    NOTEBOOKS_DIR = Path(__file__).parent.parent.resolve()

    corpus = Corpus(registry_path=NOTEBOOKS_DIR / "utils" / "article_registry.json")

    n_topics = 90
    seed = 36775

    model = Model(corpus, n_topics, seed=seed)
    model.load_topics(num_workers=10)

    viz = Visualizer(model)
    # _, (ax1) = plt.subplots(1, 1, figsize=(5, 5))

    # viz.plot_number_of_documents_per_year(model.topics[0], ax=ax1)
    # w1 = viz.plot_wordcloud_per_main_area("Value theory", ax=ax1)
    viz.plot_word_evolution_by_topic_graph(11)
    plt.savefig("example.jpg")
    plt.show()
    # viz.plot_stream_graph(ax=ax3)
