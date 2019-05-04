#!/usr/bin/env python3

import re
import logging

import pandas as pd
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
    
mti_subset_train = pd.read_csv("./data/2013_MTI_in_OA_train.csv")

# Set up logging
logging.basicConfig(filename="errors.log", level=logging.INFO,
                    filemode="w", format="%(levelname)s - %(message)s")
logger = logging.getLogger()

# List for the references
mti_refs = []

# Extract references from the XML files
for ID in tqdm(mti_subset_train['Accession ID']):
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

# PMC-ids.csv is used to convert DOIs to PMIDs, this file is available at:
# https://www.ncbi.nlm.nih.gov/pmc/pmctopmid/
pmc_ids = pd.read_csv("./data/PMC-ids.csv", low_memory=False)

# Drop unneeded columns
pmc_ids = pmc_ids.drop(["Journal Title", "ISSN", "eISSN", "Year", "Volume",
                         "Issue", "Page", "Manuscript Id", 
                         "Release Date"], axis=1)

# Change PMIDs from float64 in scientific notation to str
pmc_ids.PMID = pmc_ids.PMID.fillna(0)
pmc_ids.PMID = pmc_ids.PMID.astype(int).astype(str)
pmc_ids.PMID = pmc_ids.PMID.replace("0", "NA")

# This function converts a DOI or PMCID to a PMID
def fetch_pmid(identifier, pmc_ids, logger):
    if re.match("^10\..*$", identifier):
        pmid = pmc_ids[pmc_ids.DOI == identifier].PMID
        if not pmid.empty:
            return pmid.item()
        else:
            return np.NaN
    if re.match("^PMC.*$", identifier):
        pmid = pmc_ids[pmc_ids.PMCID == identifier].PMID
        if not pmid.empty:
            return pmid.item()
        else:
            logger.error("PMCID conversion error: {}".format(identifier))
            return identifier
    
    # Return original identifier if not a DOI or PMCID
    return identifier

# Convert IDs to PMIDs if possible
for sample in tqdm(mti_refs):
    for identifier in range(len(sample)):
        sample[identifier] = fetch_pmid(sample[identifier], pmc_ids, logger)
        

            
        
        
        
        
# Remove IDs in other formats
# add logic here to drop DOIs too, maybe just drop
# everything that isn't a PMID to make it easier
# edge_list = edge_list.replace(" 10.1007/s11606-011-1968-2", "22282311")
mti_refs = mti_refs.replace("^2-s.*$", np.NaN, regex=True)
# check backslash escape here or just replace everything tha tisn't a PMID instead
mti_refs = mti_refs.replace("^[0-9]{1,3}//..*$", np.NaN, regex=True)

# Make edge list by melting the DF. Drop unnecessary column and NAs
edge_list = pd.melt(mti_refs, id_vars=['0'], 
                    value_vars=mti_refs.loc[:, 
                    mti_refs.columns != '0'],
                    value_name='1')
edge_list = edge_list.drop("variable", axis=1)
edge_list = edge_list.dropna()

# Sort list, drop duplicates and save
edge_list = edge_list.sort_values(by=['0'])
edge_list = edge_list.drop_duplicates()
edge_list.to_csv("edge_list.csv")
