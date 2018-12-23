# Author: Kirk Gasper

from Bio import Entrez
import pandas as pd
import numpy as np
import time
import os
import re
from tqdm import tqdm
from bs4 import BeautifulSoup

os.chdir('/Users/wigasper/Documents/Research Project')

mti_oaSubset_train = pd.read_csv("2013_MTI_in_OA_train.csv")

# This method takes a DOI ID and returns a PMID if Pubmed has
# one for the DOI ID; if not, returns False
def get_PMID_from_DOI(doi_in):
    if re.match(r"^.*10\..*$", str(doi_in)):
        time.sleep(.5)
        Entrez.email = "kgasper@unomaha.edu"
        handle = Entrez.esearch(db="pubmed", term=doi_in)
        record = Entrez.read(handle)
        if len(record['IdList']) > 0:
            PMID = record['IdList'][0]
            return PMID
        elif len(record['IdList']) == 0:
            return np.NaN

#handle = Entrez.esearch(db="pubmed", term="10.1073/pnas.37.4.205")
mti_oa_short = mti_oaSubset_train

# 2D list for the references
mti_refs_short = [[]]

# This list is for IDs that don't have the 'back' tag, to investigate later.
ids_to_check = []
# FileNotFoundErrors
fnfe = []

# Extract references from the XML files
for ID in tqdm(mti_oa_short['Accession ID']):
    try:
        handle = open("./PMC XMLs/{}.xml".format(ID), "r")
        soup = BeautifulSoup(handle.read())
        
        sample = [ID]
        
        # add IDs to the error list if they don't have the 'back' tag and to 
        # the samples list if they do
        if (soup.back == None):
            ids_to_check.append(ID)
        elif (soup.back != None):
            for pubid in soup.back.find_all('pub-id'):
                sample.append(pubid.string)
            
            mti_refs_short.append(sample)
    except FileNotFoundError:
        fnfe.append(ID)
    
mti_refs_short = pd.DataFrame(mti_refs_short)

mti_refs_short.to_csv('get_refs_soup_run1.csv')

# read in if needed:
mti_refs_short = pd.read_csv("get_refs_soup_run1.csv", low_memory=False)

# this works:
for row in tqdm(range(0, len(mti_refs_short))):
    for col in range(0, len(mti_refs_short.columns)):
        if re.match(r"^.*10\..*$", str(mti_refs_short.iloc[row, col])):
            mti_refs_short.iloc[row, col] = get_PMID_from_DOI(mti_refs_short.iloc[row, col])
    
mti_refs_short.to_csv("get_refs_doi_to_pmid_run2.csv")
    

#tester = pd.read_csv("get_refs_soup_run1.csv", low_memory=False)
#tester = tester[1:250]
#
#tester = tester.applymap(lambda x: get_PMID_from_DOI(x))
#
##tester_result = mti_refs_short.replace("^.*10\..*$", lambda x: get_PMID_from_DOI(x), regex = True)
#
#mti_refs_short.to_csv("get_refs_doi_to_pmid_run1.csv")
#mti_refs_short = pd.read_csv("get_refs_doi_to_pmid_run1.csv", low_memory=False) 
#
#get_refs_doi_to_pmid_run1 = pd.read_csv("get_refs_doi_to_pmid_run1.csv", low_memory=False)
#if tester.equals(mti_refs_short[1:250]):
#    print("yes they are the same")
# last run made:
# 18%|█▊        | 1830/10329 [3:28:01<29:06:20, 12.33s/it] on 12/22 22:30
        
#repr(mti_refs_short3.iloc[[46], [2]])
#test = get_PMID_from_DOI(mti_refs_short.iloc[66, 2])
#if re.match("^.*10\..*$", mti_refs_short.iloc[66, 2]):
#    print("oh yeah")
#if not re.match("^.*10\..*$", mti_refs_short.iloc[66, 2]):
#    print("nope, not at all")



##############################################
# From first run:
# FNFE = PMC3464690
# IDs to check: PMC3197088
# PMC3284407, PMC3334562, PMC3334566, PMC3334576, PMC3334580, PMC3339691
# PMC3339694, PMC3339697, PMC3363164, PMC3366896, PMC3371872, PMC3374144
# PMC3376543, PMC3393343, PMC3398479, PMC3407656, PMC3409423, PMC3412676
# PMC3446280, PMC3446285, PMC3496406, PMC4605518, PMC4605596, PMC4605686
