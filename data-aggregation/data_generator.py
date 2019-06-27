import os
import re
import json
import argparse
import logging

def build_feature_dict(edge_list):
    term_freqs = {}
    doc_terms = {}
    with open("../data/pm_bulk_doc_term_counts.csv", "r") as handle:
        for line in handle:
            line = line.strip("\n").split(",")
            doc_terms[line[0]] = line[1:]
    
    for edge in edge_list:
        term_freqs[edge[0]] = {}
        for term in doc_terms[edge[1]]:
            if term in term_freqs[edge[0]].keys():
                term_freqs[edge[0]][term] += 1
            else:
                term_freqs[edge[0]][term] = 1
    
    for doc in term_freqs.keys():
        total_count = 0
        for term in term_freqs[doc].keys():
            total_count += term_freqs[doc][term]
        for term in term_freqs[doc].keys():
            term_freqs[doc][term] = term_freqs[doc][term] / total_count
    
    return term_freqs
    
def build_edge_list(file_list):
    article_pmid = re.compile(r'<front>.*<article-id pub-id-type="pmid">(\d+)</article-id>.*</front>')
    refs_list = re.compile(r'<back>.*<ref-list>(.*)</ref-list>.*</back>')
    ref_pmid = re.compile(r'<pub-id pub-id-type="pmid">(\d+)</pub-id>')
    
    edges = []

    for xml_file in file_list:
        if xml_file.split(".")[-1] == "nxml":
            with open(xml_file, "r") as handle:
                article = handle.readlines()

            article = "".join(article)
            
            article_id = article_pmid.search(article).group(1)
            refs = refs_list.search(article).group(1)
            
            refs = refs.split("<ref")
            for ref in refs:
                match = ref_pmid.search(ref)
                if match:
                    edges.append((article_id, match.group(1)))

    return edges

def main():
    # Get command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="A directory, can be relative to cd, containing XMLs to be parsed", type=str)
    args = parser.parse_args()
    
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("mesh_term_extraction.log")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    xmls_to_parse = os.listdir(args.input)

    xmls_to_parse = ["".join([args.input, file_name]) for file_name in xmls_to_parse]
    
    edge_list = build_edge_list(xmls_to_parse)

    with open("../data/edge_list.csv", "w") as out:
        for edge in edge_list:
            out.write("".join([edge[0], ",", edge[1], "\n"]))

    term_freqs = build_feature_dict(edge_list)

    with open("../data/term_freqs.json", "w") as out:
        json.dump(term_freqs, out)

if __name__ == "__main__":
    main()