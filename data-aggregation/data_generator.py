import os
import re
import json
import argparse
import traceback
import logging

def build_feature_dict(edge_list, logger):
    term_freqs = {}
    doc_terms = {}
    
    # Get only the docs that we need term counts for to avoid using 24 gb of memory
    need_counts_for = []
    for edge in edge_list:
        need_counts_for.append(edge[1])
    need_counts_for = set(need_counts_for)
    with open("../data/pm_bulk_doc_term_counts.csv", "r") as handle:
        for line in handle:
            line = line.strip("\n").split(",")
            if line[0] in need_counts_for:
                doc_terms[line[0]] = line[1:]
    
    term_freqs = {edge[0]: {} for edge in edge_list}
    
    for edge in edge_list:
        try:
            for term in doc_terms[edge[1]]:
                if term and term in term_freqs[edge[0]].keys():
                    term_freqs[edge[0]][term] += 1
                elif term:
                    term_freqs[edge[0]][term] = 1
        except Exception as e:
            trace = traceback.format_exc()
            logger.error(repr(e))
            logger.critical(trace)
    
    for doc in term_freqs.keys():
        total_count = 0
        for term in term_freqs[doc].keys():
            total_count += term_freqs[doc][term]
        for term in term_freqs[doc].keys():
            term_freqs[doc][term] = term_freqs[doc][term] / total_count
    
    return term_freqs
    
def build_edge_list(file_list, logger):
    article_pmid = re.compile(r'<front>.*<article-id pub-id-type="pmid">(\d+)</article-id>.*</front>')
    refs_list = re.compile(r'<back>.*<ref-list>(.*)</ref-list>.*</back>')
    ref_pmid = re.compile(r'<pub-id pub-id-type="pmid">(\d+)</pub-id>')
    
    edges = []
    from tqdm import tqdm
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

def main():
    # Get command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="A directory, can be relative to cd, containing XMLs to be parsed", type=str)
    args = parser.parse_args()
    
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("../logs/data_generator.log")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    xmls_to_parse = os.listdir(args.input)
#    xmls_to_parse = os.listdir("../pmc_xmls")

    xmls_to_parse = ["/".join([args.input, file_name]) for file_name in xmls_to_parse if file_name.split(".")[-1] == "nxml"]
#    xmls_to_parse = ["/".join(["../pmc_xmls", file_name]) for file_name in xmls_to_parse if file_name.split(".")[-1] == "nxml"]
    edge_list = build_edge_list(xmls_to_parse[0:200000], logger)

    with open("../data/edge_list.csv", "w") as out:
        for edge in edge_list:
            out.write("".join([edge[0], ",", edge[1], "\n"]))

    term_freqs = build_feature_dict(edge_list, logger)

    with open("../data/term_freqs_rev_0.json", "w") as out:
        json.dump(term_freqs, out)

if __name__ == "__main__":
    main()