import xmltodict
from Bio import Entrez
import pandas as pd
import time
import os
from tqdm import tqdm
from pathlib import Path

# OSX path:
#os.chdir('/Users/wigasper/Documents/Research Project')

# Ubuntu path:
os.chdir('/home/wkg/Documents/Research Project')

mti_train = pd.read_csv("./FUSE/2013_MTI_in_OA_train.csv")
ids_to_get = mti_train["Accession ID"].tolist()
del(mti_train)

# PMC_errors stores any errors from PMC's side
pmc_errors = []

for ID in tqdm(ids_to_get):
    start_time = time.perf_counter()
    file = Path("./MeSH XMLs/{}.xml".format(ID))
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
            file_out = open("./PMC XMLs/{}.xml".format(ID), "w")
            file_out.write(xmlString)
            
        if time.perf_counter() - start_time < .33:
            time.sleep(.33 - (time.perf_counter() - start_time))
