import os
import argparse
import logging

def build_feature_dict():
    pass
def build_edge_list(file_list):
    return 1

def main():
    # Get command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="A directory containing XMLs to be parsed", type=str)
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
if __name__ == "__main__":
    main()