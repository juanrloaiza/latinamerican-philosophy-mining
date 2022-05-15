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
            results.append(self.model.get_article_title(article_id))
        return results[:n]

    def get_top_words(self, n=10, verbose=False):
        """
        Returns a list with the top {n} words in a topic given a certain lda model.

        If verbose, it will also print the topic using gensim's LDA pretty printer.
        """

        if verbose:
            print(self.model.lda.print_topic(self.id, topn=n))

        return [
            (self.model.id2word.get(idx), f"{prob:.3f}")
            for idx, prob in self.model.lda.get_topic_terms(self.id, topn=n)
        ]

    def summary(self):
        """Returns a summary string of the topic including topic ID, top words, and top article titles."""
        articles = ["\n* " + title for title in self.get_top_titles()]
        words_table = pd.DataFrame(
            self.get_top_words(), columns=["Word", "Probability"]
        ).to_markdown()
        report = f"""# Topic {self.id}\n\n## Top words:\n{words_table}\n## Top articles:\n{''.join(articles)}\n"""

        return report
