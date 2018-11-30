#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 16:27:29 2018

@author: wigasper
"""

import xmltodict
from Bio import Entrez
import pandas as pd
import time
import os

os.chdir('/Users/wigasper/Documents/Research Project')

mti_oaSubset_train = pd.read_csv("2013_MTI_in_OA_train.csv")

# PMC_errors stores any errors from PMC's side
PMC_errors = pd.DataFrame(columns=['ID', 'code', 'message'])

for ID in mti_oaSubset_train['Accession ID']:
    Entrez.email = "kgasper@unomaha.edu"
    handle = Entrez.efetch(db="pmc", id=ID, retmode="xml")
    xmlString = handle.read()

    element = xmltodict.parse(xmlString)
    
    pmcError = False
    
    # Check for an error on PMC's side and record it
    for i in element['pmc-articleset'].keys():
        if (i == 'error'):
            refs = element['pmc-articleset']['error']
            pmcErrorData = {'ID': [ID], 'code': [refs['Code']], 'message': [refs['Message']]}
            tempPMCerrorDF = pd.DataFrame(pmcErrorData, columns=['ID', 'code', 'message'])
            PMC_errors = PMC_errors.append(tempPMCerrorDF, ignore_index=True)
            pmcError = True
    
    if not pmcError:
        file_out = open("./PMC XMLs/{}.xml".format(ID), "w")
        file_out.write(xmlString)
    
    # This is a delay in accordance with PubMed API usage guidelines.
    # It should not be set lower than .34.
    time.sleep(.4)

PMC_errors.to_csv('PMC_errors.csv')