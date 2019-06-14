#!/usr/bin/env python3
import os
import re
import time
import math
import argparse
import logging
import traceback
from multiprocessing import Process, Queue

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

def td_matrix_gen(file_path, term_subset, docs_per_matrix):
    with open("./data/pm_bulk_doc_term_counts.csv", "r") as handle:
        td_matrix = []
        for line in handle:
            if len(td_matrix) > docs_per_matrix:
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

def matrix_builder(work_queue, add_queue, id_num):
    # Set up logging - I do actually want a logger for each worker
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"./logs/term_co-occurrence_worker{id_num}.log")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    try:
        while True:
            matrix = work_queue.get()
            if matrix is None:
                break
            td_matrix = np.array(matrix)
            co_matrix = np.dot(td_matrix.transpose(), td_matrix)
            add_queue.put(co_matrix)

    except Exception as e:
        trace = traceback.format_exc()
        logger.error(repr(e))
        logger.critical(trace)

# A function for multiprocessing, pulls from the queue and writes
def matrix_adder(add_queue, completed_queue, dim, docs_per_matrix, num, logger):
    # Log counts and rates after every n matrices
    log_interval = 800
    total_processed = 0
    co_matrix = np.zeros((dim, dim))
    start_time = time.perf_counter()
    while True:
        if total_processed and total_processed % log_interval == 0:
            elapsed_time = int((time.perf_counter() - start_time) * 10) / 10.0
            time_per_it = elapsed_time / (docs_per_matrix * log_interval)
            logger.info(f"Adder {num}: {total_processed * docs_per_matrix} docs added to matrix - last batch of "
                        f"{docs_per_matrix * log_interval} at a rate of {time_per_it} sec/it")
            start_time = time.perf_counter()

        matrix_to_add = add_queue.get()
        if matrix_to_add is None:
            completed_queue.put(co_matrix)
            break
        co_matrix = co_matrix + matrix_to_add
        total_processed += 1

def main():
    # Get command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-rc", "--recount", help="recount terms for each doc", type=str)
    parser.add_argument("-n", "--num_docs", help="number of docs to make matrices for", type=int)
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

    # This value was determined in testing but is kind of arbitrary
    # Maybe need to figure out a better way to get this
    docs_per_matrix = 34

    matrix_gen = td_matrix_gen("./data/pm_bulk_doc_term_counts.csv", term_subset, docs_per_matrix)

    # Set up multiprocessing
    num_builders = 5
    num_adders = 2
    add_queue = Queue(maxsize=5)
    build_queue = Queue(maxsize=num_builders)
    completed_queue = Queue(maxsize=num_adders + 1)

    # Build co-occurrence matrices for each adder to work with
    #co_matrices = [np.zeros((len(term_subset), len(term_subset))) for _ in range(num_adders)]

    adders = [Process(target=matrix_adder, args=(add_queue, completed_queue, len(term_subset), 
                    docs_per_matrix, num, logger)) for num in range(num_adders)]
    
    for adder in adders:
        #adder.daemon = True
        adder.start()

    builders = [Process(target=matrix_builder, args=(build_queue, add_queue, num)) for num in range(num_builders)]

    for builder in builders:
        builder.start()

    if args.num_docs:
        limit = args.num_docs / docs_per_matrix
    else:
        limit = math.inf
    
    count = 0
    for matrix in matrix_gen:
        if count < limit:
            build_queue.put(matrix)
            count += 1
        else:
            break
    
    while True:
        if build_queue.empty():
            for _ in range(num_builders):
                build_queue.put(None)
            break

    for builder in builders:
        builder.join()

    for adder in adders:
        add_queue.put(None)

    if add_queue.empty():
        for adder in adders:
            adder.join()
    """
    for matrix in matrix_gen:
        start_time = time.perf_counter()
        td_matrix = np.array(matrix)
        temp_co_matrix = np.dot(td_matrix.transpose(), td_matrix)
        co_matrix = co_matrix + temp_co_matrix
        count += docs_per_matrix
        elapsed_time = time.perf_counter() - start_time
        #elapsed_time = int((time.perf_counter() - start_time) * 10) / 10.0
        time_per_it = elapsed_time / docs_per_matrix
        print(f"{count} docs added to matrix - last batch of {docs_per_matrix} at a rate of {time_per_it} sec/it")
    """
    co_matrices = []
    for _ in range(num_adders):
        co_matrices.append(completed_queue.get())

    co_matrix = sum(co_matrices)
    
    np.save("./data/co-occurrence-matrix", co_matrix)

if __name__ == "__main__":
	main()
