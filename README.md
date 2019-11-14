# FUSE
Development repo for my work on predicting PubMed MeSH terms using citation networks.

There is a lot going on here and more or less represents my sandbox during the process. Some notable files:

**baseline_expanded.ipynb** - The final simple decision tree model. Includes evaluation metrics, prediction examples, and some exploratory analyses.

**baseline_models.ipynb** - The initial work done on a reduced dataset to prove concept.

**compute_semantic_similarity.ipynb** - A notebook walking through the process to determine semantic similarity values for each term, a more palatable version of this that can run from the command line is available in the [pubmed-mesh-utils](https://github.com/wigasper/pubmed-mesh-utils) repository.

**data_aggregation_pipeline** - A notebook detailing how I aggregated data for the initial work (reduced dataset).

**data-aggregation/data_generator.py** - The script used to build the expanded dataset from NCBI's bulk data dumps.

**measure_term_co-occurrence.ipynb** - A notebok walking through the process for computing term co-occurrence log-likelihood ratios, a more palatable version of this that can run from the command line is currently in development in the [pubmed-mesh-utils](https://github.com/wigasper/pubmed-mesh-utils) repository.

More user-friendly versions of some of the tools I used here are available in the [pubmed-mesh-utils](https://github.com/wigasper/pubmed-mesh-utils) repository.
