#!/usr/bin/env python3

import os
import re

import pandas as pd
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup

mti_subset_train = pd.read_csv("./data/2013_MTI_in_OA_train.csv")

# List for the references
mti_refs = [[]]

# This list is for IDs that don't have the 'back' tag, to investigate later.
ids_to_check = []

# FileNotFoundErrors
fnfe = []

# Extract references from the XML files
for ID in tqdm(mti_subset_train['Accession ID']):
    try:
        handle = open("./PMC XMLs/{}.xml".format(ID), "r")
        soup = BeautifulSoup(handle.read())
        
        sample = [ID]
        
        # add IDs to the error list if they don't have the 'back' tag and to 
        # the samples list if they do
        if soup.back == None:
        # should i do if soup.back is None: here?
            ids_to_check.append(ID)
        elif soup.back != None:
        # if soup.back is not None: ???
            for pubid in soup.back.find_all('pub-id'):
                sample.append(pubid.string)
            
            mti_refs.append(sample)
    except FileNotFoundError:
        fnfe.append(ID)
    
mti_refs = pd.DataFrame(mti_refs)

# Read in PMC_IDs to convert all the DOIs to PMIDs:
PMC_ids = pd.read_csv("PMC-ids.csv", low_memory=False)

# Drop unneeded columns
DOI_PMIDs = PMC_ids.drop(["Journal Title", "ISSN", "eISSN", "Year", "Volume",
                         "Issue", "Page", "PMCID", "Manuscript Id", 
                         "Release Date"], axis=1)
del(PMC_ids)

# Change PMIDs from float64 in scientific notation to str
DOI_PMIDs.PMID = DOI_PMIDs.PMID.fillna(0)
DOI_PMIDs.PMID = DOI_PMIDs.PMID.astype(int).astype(str)
DOI_PMIDs.PMID = DOI_PMIDs.PMID.replace("0", "NA")

# Find DOIs and convert them to PMIDs if possible
####################################
# abstract this to a function!!!!
for row in tqdm(range(0, len(mti_refs))):
    for col in range(0, len(mti_refs.columns)):
        if re.match("^10\..*$", str(mti_refs.iloc[row, col])):
            result = DOI_PMIDs[DOI_PMIDs.DOI == mti_refs.iloc[row, col]].PMID
            if len(result) == 1:
                 mti_refs.iloc[row, col] = result.item()
            if len(result) == 0:
                mti_refs.iloc[row, col] = np.NaN
            
# Remove IDs in other formats
# add logic here to drop DOIs too, maybe just drop
# everything that isn't a PMID to make it easier
# edge_list = edge_list.replace(" 10.1007/s11606-011-1968-2", "22282311")
mti_refs = mti_refs.replace("^2-s.*$", np.NaN, regex=True)
mti_refs = mti_refs.replace("^[0-9]{1,3}[/]..*$", np.NaN, regex=True)

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
