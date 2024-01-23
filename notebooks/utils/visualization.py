"""
Implements the utilities around visualizing topics and models.
"""
from typing import Dict
from collections import Counter
import datetime as dt
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd

from utils.topic import Topic
from utils.corpus import Article
from utils.model import Model

FIG_SIZE = (10, 5)


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

    def plot_wordcloud_per_main_area(self, main_area: str):
        raise NotImplementedError()

    def plot_stream_graph(self, ax: plt.Axes = None):
        data = []
        for topic in model.topics:
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
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    viz.plot_number_of_documents_per_year(model.topics[0], ax=ax1)
    # viz.plot_wordcloud_per_main_area("Value theory", ax=ax2)
    viz.plot_stream_graph(ax=ax3)
