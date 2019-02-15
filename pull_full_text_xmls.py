import time
import os
from pathlib import Path

import xmltodict
from Bio import Entrez
import pandas as pd
from tqdm import tqdm

# OSX path:
#os.chdir('/Users/wigasper/Documents/Research Project')

# Ubuntu path:
os.chdir('/media/wkg/storage/Research Project')

mti_train = pd.read_csv("./FUSE/2013_MTI_in_OA_train.csv")
mti_test = pd.read_csv("./FUSE/2013_MTI_in_OA_test.csv")

ids_to_get = mti_train["Accession ID"].tolist() + mti_test["Accession ID"].tolist()

# PMC_errors stores any errors from PMC's side
pmc_errors = []

for ID in tqdm(ids_to_get):
    start_time = time.perf_counter()
    file = Path("./PMC XMLs/{}.xml".format(ID))
    if not file.exists():
        Entrez.email = "kgasper@unomaha.edu"
        handle = Entrez.efetch(db="pmc", id=ID, retmode="xml")
        xmlString = handle.read()
        element = xmltodict.parse(xmlString)
    
        pmc_error = False
    
        # Check for an error on PMC's side and record it
        for key in element['pmc-articleset'].keys():
            if key == 'error':
                pmc_errors.append(ID)
                pmc_error = True
    
        if not pmc_error:
            with open(file, "w") as file_out:
                file_out.write(xmlString)
            
        if time.perf_counter() - start_time < .33:
            time.sleep(.33 - (time.perf_counter() - start_time))
