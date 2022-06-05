import pandas as pd


class Topic:
    def __init__(self, id, article_list: list, model):
        """Implements a Topic class which includes some methods for each topic in a model."""
        self.id = id
        self.articles = article_list
        self.model = model

    def __repr__(self) -> str:
        """Represents itself by topic id and the number of articles it has."""
        return f"Topic: {self.id}, Articles: {len(self.articles)}"

    def __len__(self) -> int:
        """Represents the topic length by the number of articles it has."""
        return len(self.articles)

    def get_top_articles(self, n=5):
        """Returns the top n articles IDs in the topic."""
        return self.articles[:n]

    def get_top_titles(self, n=5):
        """Returns the n top article titles in the topic based on probability."""
        results = []
        for article_id, prob in self.articles:
            results.append([self.model.get_article_ref(article_id), prob])
        return results[:n]

    def get_top_words(self, n: int = 10, mass: int = None):
        """
        Returns a list with the top {n} words in a topic given a certain LDA model.

        If we pass a mass value, we return as many words until we hit the probability mass value.
        """
        if mass:
            results = []
            checked_mass = 0
            for word, prob in self.model.lda.show_topic(self.id, topn=10000):
                results.append((word, round(prob, 3)))
                checked_mass += prob
                if checked_mass >= mass:
                    break
            return results

        return [
            (word, round(prob, 3))
            for word, prob in self.model.lda.show_topic(self.id, topn=n)
        ]

    def summary(self):
        """Returns a summary string of the topic including topic ID, top words, and top article titles."""
        articles = ["\n* " + title for title in self.get_top_titles()]
        words_table = pd.DataFrame(
            self.get_top_words(), columns=["Word", "Probability"]
        ).to_markdown()
        report = f"""# Topic {self.id}\n\n## Top words:\n{words_table}\n## Top articles:\n{''.join(articles)}\n"""

        return report
