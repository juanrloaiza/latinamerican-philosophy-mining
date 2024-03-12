import pandas as pd

from utils.model import Model


class TableMaker:
    def __init__(self, model: Model) -> None:
        self.model = model

    def compute_time_slices_table(self) -> pd.DataFrame:
        time_slices_and_counts = self.model.corpus.get_slices_and_counts(
            self.model.time_window
        )

        table_rows = [
            {"decade": f"{start}-{end - 1}", "count": count}
            for (start, end), count in time_slices_and_counts.items()
        ]

        return pd.DataFrame(table_rows)

    def compute_main_area_descriptor_table(
        self,
        main_area: str,
        average_over_decades: bool = True,
        n_words: int = 10,
    ) -> pd.DataFrame:
        """
        Computes a main descriptor per area, composed of a table n_words with the probabilities of each word across all (sub)topics.
        """

        main_area_descriptor = self.model.compute_main_area_descriptor(main_area)
        if average_over_decades:
            main_area_descriptor_averaged = main_area_descriptor.mean(axis=1)
            # TODO: Change the names of columns.
            main_area_descriptor_averaged.rename(
                f"{main_area} (averaged)", inplace=True
            )

            return main_area_descriptor_averaged.sort_values(ascending=False).head(
                n_words
            )

        main_area_descriptor.sort_values(
            by=[main_area_descriptor.columns[-1]], inplace=True, ascending=False
        )
        return main_area_descriptor.head(n_words)

    def format_dataframe_as_markdown(self, df: pd.DataFrame) -> str:
        return df.to_markdown(floatfmt=".3f")

    def format_dataframe_as_latex(self, df: pd.DataFrame) -> str:
        return df.to_latex(float_format="%.3f", bold_rows=True)


if __name__ == "__main__":
    from pathlib import Path
    from utils.corpus import Corpus

    NOTEBOOKS_DIR = Path(__file__).parent.parent.resolve()

    corpus = Corpus(registry_path=NOTEBOOKS_DIR / "utils" / "article_registry.json")

    n_topics = 90
    seed = 36775

    model = Model(corpus, n_topics, seed=seed)
    model.load_topics(num_workers=5)

    table_maker = TableMaker(model)
    time_slices_table = table_maker.compute_time_slices_table()

    main_areas = model.get_main_areas().keys()

    for main_area in main_areas:
        print(main_area)
        table = table_maker.compute_main_area_descriptor_table(
            main_area=main_area, average_over_decades=True
        )
        print(table)
        print("---\n")

"""
|             |          0 |\n|:------------|-----------:|\n| moral       | 0.0213524  |\n| principio   | 0.0193511  |\n| experiencia | 0.0160565  |\n| vida        | 0.0158126  |\n| hombre      | 0.0138879  |\n| libertad    | 0.01335    |\n| juicio      | 0.0128533  |\n| humano      | 0.010545   |\n| acción      | 0.0101365  |\n| razón       | 0.00907954 |
"""
