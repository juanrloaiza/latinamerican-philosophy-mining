import sys

from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(f"{NOTEBOOKS_DIR.parent}")
sys.path.append(f"{NOTEBOOKS_DIR}")

import matplotlib.pyplot as plt
import seaborn as sns

from notebooks.utils.model import Model
from notebooks.utils.corpus import Corpus, Article


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
    doc, _ = model.topics[11].docs[4]

    # This plots a barplot with the topics of the article
    topic_idx, topic_probs = model.get_topic_distribution(doc)

    # Let's filter the first 10 topics
    _, ax = plt.subplots(1, 1)
    sns.barplot(x=topic_idx[:15].astype(str), y=topic_probs[:15], ax=ax)

    plt.show()
