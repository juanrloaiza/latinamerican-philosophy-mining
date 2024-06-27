import sys

from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(f"{NOTEBOOKS_DIR.parent}")
sys.path.append(f"{NOTEBOOKS_DIR}")

import matplotlib.pyplot as plt
import seaborn as sns

from notebooks.utils.model import Model
from notebooks.utils.corpus import Corpus, Article


def plot_topic_distribution_for_document(model, doc):
    # This plots a barplot with the topics of the article
    topic_idx, topic_probs = model.get_topic_distribution(doc)

    # Let's filter the first n topics
    n = 8
    fig, ax = plt.subplots(1, 1)
    sns.barplot(
        x=[model.get_small_topic_description(topic_id) for topic_id in topic_idx[:n]],
        y=topic_probs[:n],
        ax=ax,
    )

    # Rotate the x labels
    plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    fig.savefig("topic_for_article.jpg", dpi=300, bbox_inches="tight")

    plt.show()


def plot_word_distribution_for_topic(model, topic_id):
    topic = model.topics[topic_id]
    print(topic.word_table)
    print()
    ...


if __name__ == "__main__":
    # N_TOPICS = 10
    # base_model = Model(Corpus(registry_path="../utils/article_registry.json"), N_TOPICS)
    corpus = Corpus(registry_path=NOTEBOOKS_DIR / "utils" / "article_registry.json")

    n_topics = 90
    seed = 36775

    model = Model(corpus, n_topics, seed=seed)
    model.load_topics(num_workers=5)

    # This is an example of an interesting article to showcase
    # how LDAs and DTMs work:
    # Topic and idx: 11, 4
    # Article: El lenguaje místico y la inefabilidad
    # Author: Carlos Barbosa Cepeda
    # Date: 2016/12/01
    doc, _ = model.topics[11].docs[1]

    plot_topic_distribution_for_document(model, doc)
    plot_word_distribution_for_topic(model, 11)
