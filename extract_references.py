#!/usr/bin/env python3

import re
import logging

import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup

accessions = []

with open("./data/2013_MTI_in_OA_train_nohead.csv", "r") as handle:
    for line in handle:
        line = line.split(",")
        accessions.append(line[3])

# Set up logging
logging.basicConfig(filename="errors.log", level=logging.INFO,
                    filemode="w", format="Ref Extract: %(levelname)s - %(message)s")
logger = logging.getLogger()

# List for the references
mti_refs = []

# Extract references from the XML files
for ID in tqdm(accessions):
    try:
        with open("./PMC XMLs/{}.xml".format(ID), "r") as handle:
            soup = BeautifulSoup(handle.read())
            
            sample = [ID]
            
            # add IDs to the error log if they don't have the 'back' tag and to 
            # the samples list if they do
            if soup.back is None:
                logger.error("No refs: {}".format(str(ID)))
            elif soup.back is not None:
                for pubid in soup.back.find_all('pub-id'):
                    sample.append(pubid.string)
                mti_refs.append(sample)
    except FileNotFoundError:
        logger.error("FNFE: {}".format(str(ID)))

# Create dicts for ID conversions
dois = {}
pmcids = {}
with open("./data/PMC-ids-nohead.csv", "r") as handle:
    for line in handle:
        line = line.split(",")
        if len(line) > 9:
            if line[7]:
                dois[line[7]] = line[9]
            pmcids[line[8]] = line[9]

# This function converts a DOI or PMCID to a PMID
def fetch_pmid(identifier, dois, pmcids, logger):
    pmid = ""
    if re.match("^10\..*$", identifier):
        if identifier in dois.keys():
            pmid = dois[identifier]
        return pmid if pmid else np.NaN

    if re.match("^PMC.*$", identifier) and identifier in pmcids.keys():
        pmid = pmcids[identifier]
        if pmid:
            return pmid
        else:
            logger.error("PMCID conversion error: {}".format(identifier))
            return identifier
    
    # Return original identifier if not a DOI or PMCID
    return identifier

# Convert IDs to PMIDs if possible
for sample in mti_refs:
    for index in range(len(sample)):
        sample[index] = fetch_pmid(sample[index], dois, pmcids, logger)

edge_list = []

# Convert to edge list format and drop non-PMID identifiers:
for sample in mti_refs:
    for index in range(1, len(sample)):
        if sample[index] is not np.NaN and re.match("^\d*$", sample[index]):
            edge_list.append((sample[0], str(sample[index])))

# Remove duplicates:
edge_list = list(set(edge_list))
edge_list.sort()

# Write output
with open("./data/edge_list.csv", "w") as out:
    for edge in edge_list:
        out.write("".join([edge[0], ",", edge[1], "\n"]))
