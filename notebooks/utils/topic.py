import pandas as pd


class Topic:
    def __init__(self, id, article_list: list, model):
        self.id = id
        self.articles = article_list
        self.model = model

    def __repr__(self) -> str:
        return f"Topic: {self.id}, Articles: {len(self.articles)}"

    def get_top_articles(self, n=5):
        return self.articles[:n]

    def get_top_titles(self, n=5):
        results = []
        for article_id, prob in self.articles:
            results.append(self.model.get_article_title(article_id))
        return results[:n]

    def get_top_words(self, n=10, verbose=False):
        """
        This function returns a list with
        the top {n} words in a topic given
        a certain lda fit.

        If verbose, it will also print the
        topic using gensim's LDA pretty printer.
        """

        if verbose:
            print(self.model.lda.print_topic(self.id, topn=n))

        return [
            (self.model.id2word.get(idx), f"{prob:.3f}")
            for idx, prob in self.model.lda.get_topic_terms(self.id, topn=n)
        ]

    def summary(self):
        articles = ["\n* " + title for title in self.get_top_titles()]
        words_table = pd.DataFrame(
            self.get_top_words(), columns=["Word", "Probability"]
        ).to_markdown()
        report = f"""# Topic {self.id}\n\n## Top words:\n{words_table}\n## Top articles:\n{''.join(articles)}\n"""

        return report
