# Latin American Philosophy Mining

Authors: [Juan R. Loaiza](https://www.juanrloaiza.me) (URosario / HU Berlin) and [Miguel González Duque](https://www.miguelgondu.com) (ITU Copenhagen)

In this repository we are applying text mining to philosophy journals in Latin America.

We are starting with [Ideas y Valores](https://revistas.unal.edu.co/index.php/idval/) (Colombia) and articles from 2009 to 2017. We plan on expanding later to include more years and other journals such as [Crítica](http://critica.filosoficas.unam.mx/index.php/critica) (Mexico) and [Análisis Filosófico](https://analisisfilosofico.org/index.php/af) (Argentina).

## Structure

    .
    ├── data                # Data files (omitted from Git repository for the moment)
    |   ├── raw_html        # Raw HTML files directly as scraped with metadata     
    |   └── clean_json      # Parsed HTML files and metadata in JSON format
    ├── utils               # Helper utilities
    ├── notebooks           # Notebooks with preprocessing and analyses
    |   └── wordlists       # Stopwords and protected words lists
    └── README.md

## To-Do

* Extract view information from main HTML page.
* Calibrate the number of topics for the LDA model.
  * Implement LDA in gensim and use topic coherence measures to calibrate the number of topics.

## Preliminary figures and visualizations

![Documents by type](img/doc_by_type.png)

*Figure 1. Documents by document type.*


![Documents by type](img/doc_by_type-year.png)

*Figure 2. Documents by main type per year.*


![Most mentioned authors in the corpus](img/author_wordcloud.png)

*Figure 3. Word cloud of the most mentioned philosophers in the corpus.*

![Most frequent keywords in the corpus](img/keyword_wordcloud.png)

*Figure 4. Word cloud of the most frequent keywords in the corpus according to article metadata.*
