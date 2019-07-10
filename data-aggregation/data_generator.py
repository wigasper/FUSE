import os
import re
import time
import json
import random
import argparse
import traceback
import logging

from tqdm import tqdm

def build_feature_dict(edge_list, term_ranks, term_subset, num, logger):

    term_freqs = {}
    doc_terms = {}
    
    # Get only the docs that we need term counts for to avoid using 24 gb of memory
    need_counts_for = []
    for edge in edge_list:
        need_counts_for.append(edge[1])
    need_counts_for = set(need_counts_for)
    with open("../data/pm_doc_term_counts.csv", "r") as handle:
        for line in handle:
            line = line.strip("\n").split(",")
            if line[0] in need_counts_for:
                doc_terms[line[0]] = line[1:]
    
    term_freqs = {edge[0]: {} for edge in edge_list}
    #term_freqs = {}

    # build term freqs
    #count = 0
    #while count < len(term_freqs):
    print("Building term_freqs...")
    for edge in tqdm(edge_list):
        #if edge[0] not in term_freqs.keys():
        #    term_freqs[edge[0]] = {}
        try:
            for term in doc_terms[edge[1]]:
                if term and term in term_freqs[edge[0]].keys() and term in term_subset:
                    term_freqs[edge[0]][term] += 1
                elif term and term in term_subset:
                    term_freqs[edge[0]][term] = 1
        except Exception as e:
            trace = traceback.format_exc()
            logger.error(repr(e))
            logger.critical(trace)

    # go through term freqs and select samples
    # maybe swtich to sorting here if needed
    out = {}
    doc_count = 0
    print("Selecting samples from each threshold until maxed")
    thresholds = [x * .2 for x in range(0,5)]
    for thresh in tqdm(thresholds):
        #thresh = thresh + .2
        for doc in term_freqs.keys():
            if doc_count < num:
                sum_tot = 0
                term_count = 0
                avg = 0
                for term in term_freqs[doc].keys():
                    term_count += term_freqs[doc][term]
                    sum_tot += term_ranks[term] * term_count
                if term_count > 0:
                    avg = sum_tot / term_count
                if avg > 0 and thresh <= avg < (thresh + .2):
                    out[doc] = term_freqs[doc]
            doc_count += 1
    logger.info(f"Maxed out with {len(out)} keys")
    for doc in out.keys():
        total_count = 0
        for term in out[doc].keys():
            total_count += out[doc][term]
        for term in out[doc].keys():
            out[doc][term] = out[doc][term] / total_count

    return out
    
def build_edge_list(file_list, logger):
    article_pmid = re.compile(r'<front>.*<article-id pub-id-type="pmid">(\d+)</article-id>.*</front>')
    refs_list = re.compile(r'<back>.*<ref-list>(.*)</ref-list>.*</back>')
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
    parser.add_argument("-i", "--input", help="A directory, can be relative to cwd, containing XMLs to be parsed", type=str)
    parser.add_argument("-n", "--number", help="The number of samples to generate", type=int)
    #parser.add_argument("-m", "--minimum", help="The minimum number of times to get each sample", type=int)
    #parser.add_argument("-s", "--subset", help="Subset to only count common terms from subset list", type=str, default="../data/subset_terms_list")
    args = parser.parse_args()
    
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("../logs/data_generator.log")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Get docs list
    docs = os.listdir("../pubmed_bulk")

    # Count terms in documents
    if args.count:
        count_doc_terms(docs, logger)

    # Set up term subset and rankings
    # Rank by dividing occurrences by max to give a value between
    # 0 and 1 - allowing to choose samples with more infrequent terms
    term_subset = []
    with open("../data/subset_terms_list", "r") as handle:
        for line in handle:
            line = line.strip("\n")
            term_subset.append(line)
    term_subset = set(term_subset)

    with open("../data/pm_bulk_term_counts.json", "r") as handle:
        term_counts = json.load(handle)
    
    term_ranks = {}
    max_count = 0
    for term in term_counts.keys():
        if term in term_subset:
            term_ranks[term] = term_counts[term]
            if term_ranks[term] > max_count:
                max_count = term_ranks[term]
    
    for term in term_ranks.keys():
        term_ranks[term] = term_ranks[term] / max_count
    
    xmls_to_parse = os.listdir(args.input)

    # Shuffle the list
    random.seed(42)
    random.shuffle(xmls_to_parse)

    # did 200k samples previously
    xmls_to_parse = ["/".join([args.input, file_name]) for file_name in xmls_to_parse if file_name.split(".")[-1] == "nxml"]
    num_to_parse = args.number * 5

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
            edge_list = build_edge_list(xmls_to_parse[0:num_to_parse], logger)
        except Exception as e:
            trace = traceback.format_exc()
            logger.error(repr(e))
            logger.critical(trace)

        with open("../data/edge_list.csv", "w") as out:
            for edge in edge_list:
                out.write("".join([edge[0], ",", edge[1], "\n"]))

    #if args.minimum:
    try:
        term_freqs = build_feature_dict(edge_list, term_ranks, term_subset, args.number, logger)
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(repr(e))
        logger.critical(trace)
    #else:
    #    term_freqs = build_feature_dict(edge_list, 0, logger)

    with open("../data/term_freqs_rev_1.json", "w") as out:
        json.dump(term_freqs, out)

if __name__ == "__main__":
    main()