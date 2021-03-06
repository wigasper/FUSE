import os
import sys
import re
import time
import math
import json
import random
import argparse
import traceback
import logging

from tqdm import tqdm

def build_feature_dict(edge_list, term_ranks, term_list, num):
    logger = logging.getLogger(__name__)
    term_freqs = {}
    doc_terms = {}
    
    # Get only the docs that we need term counts for to avoid using 24 gb of memory
    # Add both edges to ensure that we are only getting samples that we have a solution
    # for - in conditional logic lines 31-34
    need_counts_for = []
    for edge in edge_list:
        need_counts_for.append(edge[0])
        need_counts_for.append(edge[1])
    need_counts_for = set(need_counts_for)
    with open("../data/pm_doc_term_counts.csv", "r") as handle:
        for line in handle:
            line = line.strip("\n").split(",")
            line = [item for item in line if item]
            if len(line) > 1 and line[0] in need_counts_for:
                doc_terms[line[0]] = line[1:]
    
    term_freqs = {edge[0]: {} for edge in edge_list if edge[0] in doc_terms.keys()}
    
    # subset edge list for only edge[0]s we have solutions for, avoid KeyErrors
    logger.info(f"edge_list len before subset: {len(edge_list)}")
    edge_list = [edge for edge in edge_list if edge[0] in term_freqs.keys()]
    logger.info(f"edge_list len after subset: {len(edge_list)}")

    # build term freqs
    logger.info(f"Building term_freqs... Pre-build: {len(term_freqs)} docs")
    for edge in tqdm(edge_list):
        try:
            if edge[1] in doc_terms.keys():
                for term in doc_terms[edge[1]]:
                    if term and term in term_freqs[edge[0]].keys() and term in term_list:
                        term_freqs[edge[0]][term] += 1
                    elif term and term in term_list:
                        term_freqs[edge[0]][term] = 1
        except Exception as e:
            trace = traceback.format_exc()
            logger.error(repr(e))
            logger.critical(trace)

    logger.info(f"{len(term_freqs)} docs added to term_freqs")
    
    for doc in term_freqs.keys():
        total_count = 0
        for term in term_freqs[doc].keys():
            total_count += term_freqs[doc][term]
        for term in term_freqs[doc].keys():
            term_freqs[doc][term] = term_freqs[doc][term] / total_count
    
    # go through term freqs and select samples
    # maybe switch to sorting here if needed

    out = {}
    doc_count = 0
    logger.info("Selecting samples from each threshold until maxed...")
    thresholds = [x * .1 for x in range(0,10)]
    for thresh in tqdm(thresholds):
        for doc in term_freqs.keys():
            if doc_count < num:
                sum_tot = 0
                term_count = 0
                avg = 0
                for term in term_freqs[doc].keys():
                    term_count += 1
                    if term in term_ranks.keys():
                        sum_tot += term_ranks[term] * term_freqs[doc][term]
                if term_count > 0:
                    avg = sum_tot / term_count
                if avg > 0 and thresh <= avg < (thresh + .1):
                    out[doc] = term_freqs[doc]
                    doc_count += 1
    logger.info(f"Selected {len(out)} keys from a pool of {len(term_freqs)}")
    
    return out
    
def build_edge_list(file_list, logger):
    article_pmid = re.compile(r'<front>[\s\S]*<article-id pub-id-type="pmid">(\d+)</article-id>[\s\S]*</front>')
    refs_list = re.compile(r'<back>[\s\S]*<ref-list([\s\S]*)</ref-list>[\s\S]*</back>')
    ref_pmid = re.compile(r'<pub-id pub-id-type="pmid">(\d+)</pub-id>')
    
    edges = []
    print("Building edge list")
    logger.info("Starting edge list build")
    for xml_file in tqdm(file_list):
        try:
            #if xml_file.split(".")[-1] == "nxml":
            with open(xml_file, "r") as handle:
                article = handle.readlines()

            article = "".join(article)
            
            article_id_match = article_pmid.search(article)
            refs_match = refs_list.search(article)

            if article_id_match and refs_match:
                article_id = article_id_match.group(1)
                refs = refs_match.group(1)
            
                refs = refs.split("<ref")
                for ref in refs:
                    ref_match = ref_pmid.search(ref)
                    if ref_match:
                        edges.append((article_id, ref_match.group(1)))
        except Exception as e:
            trace = traceback.format_exc()
            logger.error(repr(e))
            logger.critical(trace)

    return edges

def count_doc_terms(doc_list, logger):
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
            with open(f"../pubmed_bulk/{doc}", "r") as handle:
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
                                    if mesh_match and mesh_match.group(1):
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

    with open("../data/pm_doc_term_counts.csv", "w") as out:
        for doc in doc_terms:
            out.write("".join([doc, ","]))
            out.write(",".join(doc_terms[doc]))
            out.write("\n")


def main():
    # Get command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--count", help="Count term occurrences from corpus", action="store_true")
    parser.add_argument("-e", "--edges", help="Edge list path - will build list if no arg provided", type=str)
    parser.add_argument("-i", "--input", help="A directory, can be relative to cwd, containing XMLs to build an edge list from", type=str)
    parser.add_argument("-n", "--number", help="The number of samples to generate", type=int)
    #parser.add_argument("-m", "--minimum", help="The minimum number of times to get each sample", type=int)
    parser.add_argument("-s", "--subset", help="A list of MeSH term IDs to include in the data set (allows for subsetting)." \
                        "Uses all MeSH terms by default.", type=str)
    args = parser.parse_args()
    
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("../logs/data_generator.log")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Count terms in documents
    if args.count:
        docs = os.listdir("../pubmed_bulk")
        count_doc_terms(docs, logger)

    # Set up term subset and rankings
    # Rank by dividing occurrences by max to give a value between
    # 0 and 1 - allowing to choose samples with more infrequent terms
    term_list = []
    if args.subset:
        with open(args.subset, "r") as handle:
            for line in handle:
                line = line.strip("\n")
                term_list.append(line)
    else:
        with open("../data/mesh_data.tab", "r") as handle:
            for line in handle:
                line = line.strip("\n").split("\t")
                term_list.append(line[0])
    term_list = set(term_list)

    with open("../data/pm_bulk_term_counts.json", "r") as handle:
        term_counts = json.load(handle)
    
    term_ranks = {}
    max_count = 0
    for term in term_counts.keys():
        if term in term_list and term_counts[term] > 0:
            term_ranks[term] = math.log(term_counts[term])
            if term_ranks[term] > max_count:
                max_count = term_ranks[term]
    
    for term in term_ranks.keys():
        term_ranks[term] = term_ranks[term] / max_count

    #####################
    # Because of htis logic need to put something here in case that
    # it runs out of samples before num is reached
    if args.edges:
        edge_list = []
        with open(args.edges, "r") as handle:
            for line in handle:
                line = line.strip("\n").split(",")
                edge_list.append([line[0], line[1]])
    else:
        try:
            # TODO: add something here to raise an exception if no args.input
            #xmls_to_parse = os.listdir("../pmc_xmls")
            xmls_to_parse = os.listdir(args.input)
            xmls_to_parse = list(dict.fromkeys(xmls_to_parse))

            # Shuffle the list
            random.seed(42)
            random.shuffle(xmls_to_parse)

            xmls_to_parse = ["/".join([args.input, file_name]) for file_name in xmls_to_parse if file_name.split(".")[-1] == "nxml"]
            num_to_parse = args.number * 5

            edge_list = build_edge_list(xmls_to_parse[0:num_to_parse], logger)
        except Exception as e:
            trace = traceback.format_exc()
            logger.error(repr(e))
            logger.critical(trace)

        with open("../data/edge_list_build_aug7.csv", "w") as out:
            for edge in edge_list:
                out.write("".join([edge[0], ",", edge[1], "\n"]))

    try:
        term_freqs = build_feature_dict(edge_list, term_ranks, term_list, args.number)
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(repr(e))
        logger.critical(trace)

    with open("../data/term_freqs_rev_4_all_terms.json", "w") as out:
        json.dump(term_freqs, out)

if __name__ == "__main__":
    main()
