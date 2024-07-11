from __future__ import annotations

import sys

from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(f"{NOTEBOOKS_DIR.parent}")
sys.path.append(f"{NOTEBOOKS_DIR}")

ROOT_DIR = NOTEBOOKS_DIR.parent

import matplotlib.pyplot as plt
import seaborn as sns

from notebooks.utils.model import Model
from notebooks.utils.corpus import Corpus, Article
from notebooks.utils.visualization import Visualizer


def plot_topic_distribution_for_document(
    model: Model, doc, max_topics: int = 8
) -> list[int]:
    # This plots a barplot with the topics of the article
    topic_idx, topic_probs = model.get_topic_distribution(doc)

    # Let's filter the first n topics
    fig, ax = plt.subplots(1, 1)
    sns.barplot(
        x=[
            model.get_small_topic_description(topic_id)
            for topic_id in topic_idx[:max_topics]
        ],
        y=topic_probs[:max_topics],
        ax=ax,
    )

    # Rotate the x labels
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig(
        ROOT_DIR / "data" / "reports" / "topic_for_article.jpg",
        dpi=300,
        bbox_inches="tight",
    )

    return topic_idx[:max_topics]


def plot_word_tables_for_topics(model, topic_idxs: list[int]):
    for topic_id in topic_idxs:
        topic = model.topics[topic_id]
        word_table = topic.word_table

        for j, column in enumerate(word_table.columns):
            all_words_in_decade = word_table.iloc[:, j]

            # Sorting the words by probability
            all_words_in_decade = all_words_in_decade.sort_values(ascending=False)

            # Saving them to a file
            all_words_in_decade.to_csv(
                ROOT_DIR
                / "data"
                / "reports"
                / f"top_words_in_decade_{str(column)}_for_topic_{model.get_small_topic_description(topic_id)}.csv"
            )

        # print("topic_id:", topic_id, model.get_small_topic_description(topic_id))
        # df = topic.top_word_evolution_table()
        # print(topic.top_word_evolution_table())

        # # Let's save a file where we separate the top words by time slice
        # # including the probability of the word in that time slice.

        # for column in topic.word_table.columns:
        #     print(column, topic.word_table[column].values[:10])


if __name__ == "__main__":
    # N_TOPICS = 10
    # base_model = Model(Corpus(registry_path="../utils/article_registry.json"), N_TOPICS)
    corpus = Corpus(registry_path=NOTEBOOKS_DIR / "utils" / "article_registry.json")

    n_topics = 90
    seed = 36775

    model = Model(corpus, n_topics, seed=seed)
    model.load_topics(num_workers=5)

    viz = Visualizer(model)

    # This is an example of an interesting article to showcase
    # how LDAs and DTMs work:
    # Topic and idx: 11, 1
    # Article: ¿Es Wittgetstein un fundacionalista?
    # Author: Carlos Alberto Cardona S.
    # Date: 2011/05/01
    doc, _ = model.topics[11].docs[1]

    topic_idxs = plot_topic_distribution_for_document(model, doc, max_topics=8)
    _, axes = plt.subplots(3, 1, sharex=True)
    for topic_idx, ax in zip(topic_idxs, axes):
        viz.plot_word_evolution_by_topic_graph(topic_idx, ax=ax)
    plot_word_tables_for_topics(model, topic_idxs)
    plt.show()
