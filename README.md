# FUSE
Development repo for my work on predicting PubMed MeSH terms using citation networks.

There is a lot going on here and more or less represents my sandbox during the process. Some notable files:

**baseline_expanded.ipynb** - the final simple decision tree model

**baseline_models.ipynb** - the initial work done on a reduced dataset to prove concept

**compute_semantic_similarity.ipynb** - a notebook walking through the process to determine semantic similarity values for each term, a more palatable version of this that can run from the command line is available in the [pubmed-mesh-utils](https://github.com/wigasper/pubmed-mesh-utils) repository

**data_aggregation_pipeline** - a notebook detailing how I aggregated data for the initial work (reduced dataset)

**data-aggregation/data_generator.py** - the script used to build the expanded dataset from NCBI's bulk data dumps

**measure_term_co-occurrence.ipynb** - a notebok walking through the process for computing term co-occurrence log-likelihood ratios, a more palatable version of this that can run from the command line is currently in development in the [pubmed-mesh-utils](https://github.com/wigasper/pubmed-mesh-utils) repository

More user-friendly versions of some of the tools I used here are available in the [pubmed-mesh-utils](https://github.com/wigasper/pubmed-mesh-utils) repository.
