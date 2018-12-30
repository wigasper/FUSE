# Author: Kirk Gasper

#from Bio import Entrez
import pandas as pd
import numpy as np
#import time
import os
import re
from tqdm import tqdm
from bs4 import BeautifulSoup

os.chdir('/Users/wigasper/Documents/Research Project')

mti_oaSubset_train = pd.read_csv("2013_MTI_in_OA_train.csv")

# List for the references
mti_refs = [[]]

# This list is for IDs that don't have the 'back' tag, to investigate later.
ids_to_check = []

# FileNotFoundErrors
fnfe = []

# Extract references from the XML files 8177
for ID in tqdm(mti_oaSubset_train['Accession ID']):
    try:
        handle = open("./PMC XMLs/{}.xml".format(ID), "r")
        soup = BeautifulSoup(handle.read())
        
        sample = [ID]
        
        # add IDs to the error list if they don't have the 'back' tag and to 
        # the samples list if they do
        if soup.back == None:
            ids_to_check.append(ID)
        elif soup.back != None:
            for pubid in soup.back.find_all('pub-id'):
                sample.append(pubid.string)
            
            mti_refs.append(sample)
    except FileNotFoundError:
        fnfe.append(ID)
    
mti_refs = pd.DataFrame(mti_refs)

# Save refs
# mti_refs.to_csv('mti_refs_w_DOIs.csv')

# Read in if needed
mti_refs = pd.read_csv("mti_refs_w_DOIs.csv", low_memory=False)

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


## 10.1111/j.1365-4362.1994.tb02872.x
#result = DOI_PMIDs[DOI_PMIDs.DOI == "10.1111/j.1365-4362.1994.tb02872.x"].PMID
#result = DOI_PMIDs[DOI_PMIDs.DOI == "10.1016/j.ijid.2008.06.019"].PMID

# Find DOIs and convert them to PMIDs if possible. check col 4/93
for row in tqdm(range(0, len(mti_refs))):
    for col in range(0, len(mti_refs.columns)):
        if re.match(r"^[1][0][.]..*$", str(mti_refs.iloc[row, col])):
            result = DOI_PMIDs[DOI_PMIDs.DOI == mti_refs.iloc[row, col]].PMID
            if len(result) == 1:
                 mti_refs.iloc[row, col] = result.item()
            if len(result) == 0:
                mti_refs.iloc[row, col] = np.NaN

# Save if needed:
mti_refs.to_csv("mti_refs_wo_DOIs.csv")

# Read in if needed:
# mti_refs = pd.read_csv("mti_refs_wo_DOIs.csv", low_memory=False)
# mti_refs = mti_refs.drop("Unnamed: 0", axis=1)
            
# Remove IDs in the format "2-s......." 3499
mti_refs = mti_refs.replace("^2-s..*$", np.NaN, regex=True)

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

#
#get_refs_doi_to_pmid_run1 = pd.read_csv("get_refs_doi_to_pmid_run1.csv", low_memory=False)
#if tester.equals(mti_refs_short[1:250]):
#    print("yes they are the same")
# last run made:
# 18%|█▊        | 1830/10329 [3:28:01<29:06:20, 12.33s/it] on 12/22 22:30
        
#repr(mti_refs_short3.iloc[[46], [2]])
#test = get_PMID_from_DOI(mti_refs_short.iloc[66, 2])
if re.match("^[2][-][s]..*$", "2-s2.0-77954665047"):
    print("oh yeah")
if not re.match("^[2][-][s]..*$", "2-s2.0-77954665047"):
    print("nope, not at all")

if re.match("^[1][0][.]..*$", "10.1006/pmed.2000.0731.S009174350090731X"):
    print("oh yeah")
if not re.match("^[1][0][.]..*$", "10.1006/pmed.2000.0731.S009174350090731X"):
    print("nope, not at all")
    
if re.match("^..*$", "10.1006/pmed.2000.0731.S009174350090731X"):
    print("oh yeah")
if not re.match("^.*10\..*$", "10.1006/pmed.2000.0731.S009174350090731X"):
    print("nope, not at all")


##############################################
# From first run:
# IDs to check: PMC3197088
# PMC3284407, PMC3334562, PMC3334566, PMC3334576, PMC3334580, PMC3339691
# PMC3339694, PMC3339697, PMC3363164, PMC3366896, PMC3371872, PMC3374144
# PMC3376543, PMC3393343, PMC3398479, PMC3407656, PMC3409423, PMC3412676
# PMC3446280, PMC3446285, PMC3496406, PMC4605518, PMC4605596, PMC4605686
