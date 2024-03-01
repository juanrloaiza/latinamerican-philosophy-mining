"""General tests for the project.

This scripts
- verifies that the RNG seed indeed produces the same results.

"""

from pathlib import Path

from multiprocessing import Pool

from utils.model import Model
from utils.corpus import Corpus

THIS_DIR = Path(__file__).parent.resolve()


def train_model_w_fixed_seed(path):
    corpus = Corpus(registry_path=THIS_DIR / "utils" / "article_registry.json")
    model = Model(corpus, 90, time_window=10, seed=1, custom_output_path=path)
    model.prepare_corpus()
    model.train()


if __name__ == "__main__":
    # Training both models at the same time, using a fixed seed.
    paths = [
        THIS_DIR / "models" / "seed_1_example1",
        THIS_DIR / "models" / "seed_1_example2",
    ]
    with Pool(2) as p:
        p.map(train_model_w_fixed_seed, paths)
