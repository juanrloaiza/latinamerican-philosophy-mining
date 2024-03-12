"""
Implements the utilities around visualizing topics and models.

TODO:
- A sort of pie-chart for each main area detailing
  the contribution of each area (proportion: n documents).
"""

from typing import Dict, List, Tuple
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

# Some helper functions


def count_docs_in_list_of_topics_per_year(topics: List[Topic]) -> List[Tuple[int, int]]:
    """
    Counts number of documents per year in a list of topics.
    Returns a sorted list of tuples (year, count)
    """

    document_count_per_year_in_list = defaultdict(int)
    for topic in topics:
        if topic.is_trash:
            continue

        docs = [d for d, _ in topic.docs]
        for d in docs:
            document_count_per_year_in_list[d.get_year()] += 1

    # Consider years where n_docs = 0
    for year in range(
        min(document_count_per_year_in_list), max(document_count_per_year_in_list)
    ):
        if year in document_count_per_year_in_list:
            continue

        document_count_per_year_in_list[year] = 0

    return sorted(document_count_per_year_in_list.items())


class Visualizer:
    def __init__(self, model: Model) -> None:
        self.model = model

    def plot_number_of_documents_per_year_in_topic(
        self, topic_id: int, ax: plt.Axes = None
    ) -> plt.Axes:
        """Plots the number of documents per year in a given topic."""

        topic = self.model.topics[topic_id]

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=FIG_SIZE)

        counts = topic.count_documents_per_year()
        ax.bar(counts.keys(), counts.values())
        ax.set_xlim(1950, 2025)  # TODO: Should these values be hard coded?

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

        ax.set_title(main_area)
        ax.imshow(wordcloud)

    def plot_stream_graph(self, ax: plt.Axes = None, normalized: bool = False):
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
        df = df.groupby(["Date", "Main area"]).size().unstack()

        if normalized:
            df = df.div(df.sum(axis=1), axis=0)
            df.plot(kind="bar", stacked=True, ax=ax, width=0.9)
        else:
            df.plot(kind="area", stacked=True, ax=ax)

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

    def plot_streamgraph_main_and_subarea(self, main_area: str, ax: plt.Axes = None):
        # Computing the main areas from the topics inside the model.
        topics_in_main_area = self.model.get_main_areas()[main_area.capitalize()]

        # Compute the total mass of the main area.
        sorted_docs_per_year_in_main_area = count_docs_in_list_of_topics_per_year(
            topics_in_main_area
        )

        x_main_area = [x for x, _ in sorted_docs_per_year_in_main_area]
        y_main_area = [y for _, y in sorted_docs_per_year_in_main_area]

        # Get biggest subarea in the main area
        total_doc_count_in_main_area = self.model.count_docs_per_main_area(main_area)

        largest_subarea = sorted(
            total_doc_count_in_main_area.items(), key=lambda d: d[1], reverse=True
        )[0][0]

        topics_in_largest_subarea = set()
        for topic in self.model.topics:
            if topic.is_trash:
                continue

            if topic.main_area.capitalize() != main_area.capitalize():
                continue

            if largest_subarea in topic.areas:  # Check for capitalization!
                topics_in_largest_subarea.add(topic)

        topics_in_largest_subarea = list(topics_in_largest_subarea)

        document_count_per_year_in_subarea = count_docs_in_list_of_topics_per_year(
            topics_in_largest_subarea
        )

        for x in x_main_area:
            if x not in [x for x, _ in document_count_per_year_in_subarea]:
                document_count_per_year_in_subarea.append((x, 0))

        document_count_per_year_in_subarea = sorted(
            document_count_per_year_in_subarea, key=lambda d: d[0]
        )

        x_subarea = [x for x, _ in document_count_per_year_in_subarea]
        y_subarea = [y for _, y in document_count_per_year_in_subarea]

        # Plot

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=FIG_SIZE)

        ax.set_title(main_area)
        ax.stackplot(x_main_area, y_main_area, baseline="sym", colors=["#aaa"])
        ax.stackplot(
            x_subarea,
            y_subarea,
            baseline="sym",
            labels=[largest_subarea],
            alpha=0.5,
            colors=["blue"],
        )

        ax.legend()


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

    main_areas = list(model.get_main_areas().keys())

    viz.plot_streamgraph_main_and_subarea("Philosophical traditions")
    # _, (ax1) = plt.subplots(1, 1, figsize=(5, 5))

    # viz.plot_number_of_documents_per_year(model.topics[0], ax=ax1)
    # w1 = viz.plot_wordcloud_per_main_area("Value theory", ax=ax1)
    # viz.plot_word_evolution_by_topic_graph(11)
    # plt.savefig("example.jpg")
    plt.show()
    # viz.plot_stream_graph(ax=ax3)
