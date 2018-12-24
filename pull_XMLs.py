import xmltodict
from Bio import Entrez
import pandas as pd
import time
import os
from tqdm import tqdm

os.chdir('/Users/wigasper/Documents/Research Project')

mti_oaSubset_train = pd.read_csv("2013_MTI_in_OA_train.csv")

ids_to_get = mti_oaSubset_train[9001:10353]

# PMC_errors stores any errors from PMC's side
PMC_errors = pd.DataFrame(columns=['ID', 'code', 'message'])

for ID in tqdm(ids_to_get['Accession ID']):
    Entrez.email = "kgasper@unomaha.edu"
    handle = Entrez.efetch(db="pmc", id=ID, retmode="xml")
    xmlString = handle.read()
    element = xmltodict.parse(xmlString)

    pmc_error = False

    # Check for an error on PMC's side and record it
    for key in element['pmc-articleset'].keys():
        if key == 'error':
            refs = element['pmc-articleset']['error']
            pmcErrorData = {'ID': [ID], 'code': [refs['Code']],
                            'message': [refs['Message']]}
            tempPMCerrorDF = pd.DataFrame(pmcErrorData, columns=['ID', 'code',
                                                                 'message'])
            PMC_errors = PMC_errors.append(tempPMCerrorDF, ignore_index=True)
            pmc_error = True

    if not pmc_error:
        file_out = open("./PMC XMLs/{}.xml".format(ID), "w")
        file_out.write(xmlString)

    # This is a delay in accordance with PubMed API usage guidelines.
    # It should not be set lower than .34.
    time.sleep(.4)

PMC_errors.to_csv('PMC_errors_run3.csv')
