import time
import logging
from pathlib import Path

import xmltodict
from Bio import Entrez
import pandas as pd
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("pmc_api_pull.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
with open("../ncbi.key") as handle:
    api_key = handle.read()

# Add source for oa_file_list here
oa_list = pd.read_csv("../data/oa_file_list.csv")

# Subset the 2013 MTI dataset for only those PMIDs that
# are also in the PMC Open Access file list
with open("../data/PMIDs_train", "r") as fp:
    mti_train = fp.readlines()
    mti_train = pd.DataFrame({'PMID':mti_train})

with open("../data/PMIDs_test", "r") as fp:
    mti_test = fp.readlines()
    mti_test = pd.DataFrame({'PMID':mti_test})

mti_subset_train = oa_list[(oa_list.PMID.isin(mti_train.PMID))]
mti_subset_train.to_csv("../data/2013_MTI_in_OA_train.csv")

mti_subset_test = oa_list[(oa_list.PMID.isin(mti_test.PMID))]
mti_subset_test.to_csv("../data/2013_MTI_in_OA_test.csv")

ids_to_get = mti_subset_train["Accession ID"].tolist() + mti_subset_test["Accession ID"].tolist()

# Save full texts for each PMC ID
for pmcid in tqdm(ids_to_get):
    start_time = time.perf_counter()
    file = Path(f"../pmc_xmls/{pmcid}.xml")
    if not file.exists():
        Entrez.email = "kgasper@unomaha.edu"
        Entrez.api_key = api_key
        handle = Entrez.efetch(db="pmc", id=pmcid, retmode="xml")
        xmlString = handle.read()
        element = xmltodict.parse(xmlString)
    
        pmc_error = False
    
        # Check for an error on PMC's side and record it
        for key in element['pmc-articleset'].keys():
            if key == 'error':
                logger.error(f"PMC API error - ID: {pmcid}")
                pmc_error = True
    
        if not pmc_error:
            with open(file, "w") as file_out:
                file_out.write(xmlString)
            
        if time.perf_counter() - start_time < .1:
            time.sleep(.1 - (time.perf_counter() - start_time))
