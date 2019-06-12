#!/usr/bin/env python3
import os
import re
import time
import argparse
import logging
import traceback

import numpy as np
from tqdm import tqdm

def count_doc_terms(doc_list, term_subset, logger):
    doc_terms = {}
    doc_pmid = ""
    term_ids = []

    # Compile regexes
    pm_article_start = re.compile(r"\s*<PubmedArticle>")
    pm_article_stop = re.compile(r"\s*</PubmedArticle>")
    pmid = re.compile(r"\s*<PMID.*>(\d*)</PMID>")
    mesh_list_start = re.compile(r"\s*<MeshHeadingList>")
    mesh_list_stop = re.compile(r"\s*</MeshHeadingList>")
    mesh_term_id = re.compile(r'\s*<DescriptorName UI="(D\d+)".*>')

    logger.info("Starting doc/term counting")
    for doc in tqdm(doc_list):
        try:
            with open(f"./pubmed_bulk/{doc}", "r") as handle:
                start_doc_count = len(doc_terms.keys())
                start_time = time.perf_counter()

                line = handle.readline()
                while line:
                    if pm_article_start.search(line):
                        if doc_pmid:
                            doc_terms[doc_pmid] = term_ids
                            doc_pmid = ""
                            term_ids = []
                        while not pm_article_stop.search(line):
                            if not doc_pmid and pmid.search(line):
                                doc_pmid = pmid.search(line).group(1)
                            if mesh_list_start.search(line):
                                while not mesh_list_stop.search(line):
                                    mesh_match = mesh_term_id.search(line)
                                    if mesh_match and mesh_match.group(1) in term_subset:
                                        term_ids.append(mesh_match.group(1))
                                    line = handle.readline()
                            line = handle.readline()
                    line = handle.readline()
                doc_terms[doc_pmid] = term_ids

                # Get count for log
                docs_counted = len(doc_terms.keys()) - start_doc_count
                # Get elapsed time and truncate for log
                elapsed_time = int((time.perf_counter() - start_time) * 10) / 10.0
                logger.info(f"{doc} parsing completed - terms extracted for {docs_counted} documents in {elapsed_time} seconds")
                
        except Exception as e:
            trace = traceback.format_exc()
            logger.error(repr(e))
            logger.critical(trace)

    logger.info("Stopping doc/term counting")

    with open("./data/pm_bulk_doc_term_counts.csv", "w") as out:
        for doc in doc_terms:
            out.write("".join([doc, ","]))
            out.write(",".join(doc_terms[doc]))
            out.write("\n")

def td_matrix_gen(file_path, term_subset):
    with open("./data/pm_bulk_doc_term_counts.csv", "r") as handle:
        td_matrix = []
        for line in handle:
            if len(td_matrix) > 1000:
                yield td_matrix
                td_matrix = []
            line = line.strip("\n").split(",")
            terms = line[1:]
            row = []
            for uid in term_subset:
                if uid in terms:
                    row.append(1)
                else:
                    row.append(0)
            td_matrix.append(row)

def main():
    # Get command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-rc", "--recount", help="recount terms for each doc", type=str)
    args = parser.parse_args()

    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("./logs/term_co-occurrence.log")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Load term subset to count for
    term_subset = []
    with open("./data/subset_terms_list", "r") as handle:
        for line in handle:
            term_subset.append(line.strip("\n"))
    term_subset = set(term_subset)

    if args.recount:
        docs = os.listdir("./pubmed_bulk")
        count_doc_terms(docs, term_subset, logger)

    matrix_gen = td_matrix_gen("./data/pm_bulk_doc_term_counts.csv", term_subset)

    td_matrix = np.array(next(matrix_gen))
    co_matrix = np.dot(td_matrix.transpose(), td_matrix)

if __name__ == "__main__":
	main()
