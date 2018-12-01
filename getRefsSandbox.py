#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 12:21:45 2018

@author: wigasper
"""

# PMC2829485 has 'mixed-citation'

import xmltodict
from Bio import Entrez
import pandas as pd
import time
import os
from tqdm import tqdm

os.chdir('/Users/wigasper/Documents/Research Project')

mti_oaSubset_train = pd.read_csv("2013_MTI_in_OA_train.csv")

# This method takes a DOI ID and returns a PMID and True if Pubmed has
# one for the DOI ID; if not, returns False
def getPMIDfromDOI(doiIn):
    time.sleep(.34)
    handle = Entrez.esearch(db="pubmed", term=doiIn)
    record = Entrez.read(handle)
    if (len(record['IdList']) > 0):
        PMID = record['IdList'][0]
        return PMID, True
    else:
        return False

# This top section is used to pull single articles for examination

randomPMID = "PMC2829485"
Entrez.email = "kgasper@unomaha.edu"
handle = Entrez.efetch(db="pmc", id=randomPMID, retmode="xml")
xmlString = handle.read()

element = xmltodict.parse(xmlString)

############################################
# This is the testing area for pulling the references for multiple articles
# and putting them all into a data frame together.

# TO DO:
# there can be combinations of ordereddicts and lists in references
# for the citation tag. 
# need to do logging with python's logger, not these DFs
# sometimes pub-id is a list!
# currently keyError:0 with PMC2829485
# mixed-citation doi-getting is working but recording 'True' in the DF

# error_log stores tracked exceptions for analysis later. Entries
# may need to be removed from the final data set.
#error_log = pd.DataFrame(columns=['ID', 'error'])

mti_oa_short = mti_oaSubset_train[0:500]
mti_refs_short = pd.DataFrame()

for ID in tqdm(mti_oa_short['Accession ID']):
#    Entrez.email = "kgasper@unomaha.edu"
#    handle = Entrez.efetch(db="pmc", id=ID, retmode="xml")
    handle = open("./PMC XMLs/{}.xml".format(ID), "r")
    xmlString = handle.read()
    
    element = xmltodict.parse(xmlString)
    
    # These boolean values describe whether or XML tags exist in 
    # an article. With the current logic it is needed to prevent KeyErrors.
    pmcError = False
    hasRefList = False
    hasElementCitation = False
    hasCitation = False
    hasPMID = False
    hasDOI = False
    
    # Check for an error on PMC's side
#    for i in element['pmc-articleset'].keys():
#        if (i == 'error'):
#            refs = element['pmc-articleset']['error']
#            pmcErrorData = {'ID': [ID], 'code': [refs['Code']], 'message': [refs['Message']]}
#            tempPMCerrorDF = pd.DataFrame(pmcErrorData, columns=['ID', 'code', 'message'])
#            PMC_errors = PMC_errors.append(tempPMCerrorDF, ignore_index=True)
#            pmcError = True
    
    # Check for a reference list, assign it, and set hasRefList true.
    for j in element['pmc-articleset']['article'].keys():
        if (j == 'back'):
            for i in element['pmc-articleset']['article']['back'].keys():
                if (i == 'ref-list'):
                    refs = element['pmc-articleset']['article']['back']['ref-list']['ref']
                    hasRefList = True

    # TO DO:
    # need to put some logging here if an article doesn't have 'ref-list'
    # record which articles don't have it for examination

    # evaluating structure like this may not be ideal. some articles
    # have ref lists with citations that are ordereddicts and lists
#    if hasRefList:
#        if isinstance(refs, list):
#            for i in refs[0].keys():
#                if (i == 'citation'):
#                    hasCitation = True
#                if (i == 'element-citation'):
#                    hasElementCitation = True
        

    # This is a temporary data frame used to hold all the reference IDs for
    # one article. IDtype may be unnecessary.
    refIDs = pd.DataFrame(columns=['ID', 'IDtype'])
    if isinstance(refs, list) and hasRefList:
        for ref in range(0, len(refs)):
            for i in refs[ref].keys():
                if (i == 'citation'):
                    if (isinstance(refs[ref]['citation'], dict)):
                        for k in refs[ref]['citation'].keys():
                            if (k == 'pub-id'):
                                tempData = {'ID': [refs[ref]['citation']['pub-id']['#text']],
                                           'IDtype':
                                               [refs[ref]['citation']['pub-id']['@pub-id-type']]}
                                tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                                refIDs = refIDs.append(tempDF, ignore_index=True)
                    if (isinstance(refs[ref]['citation'], list)):
                        for m in range(0, len(refs[ref]['citation'])):
                            for n in refs[ref]['citation'][m].keys():
                                if (n == 'pub-id'):
                                    # this section needs to be modified to work like
                                    # the mixed-citation section
                                    if (refs[ref]['citation'][m]['pub-id']['@pub-id-type'] 
                                    == 'doi'):
#                                        hasDOI = True
#                                        doiIndex = m
                                        if not (getPMIDfromDOI(refs[ref]['citation'][m]['pub-id']['#text'])):
                                            pass
                                            #tempErrorData = {'ID': [ID], 'error': ['no_pmid']}
                                            #tempErrorDF = pd.DataFrame(tempData, columns=['ID', 'error'])
                                            #error_log = error_log.append(tempDF, ignore_index=True)
                                        else:
                                            tempData = {'ID': [getPMIDfromDOI(refs[ref]['citation'][m]['pub-id']['#text'])],
                                                       'IDtype': ['PMID']}
                                            tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                                    if (refs[ref]['citation'][m]['pub-id']['@pub-id-type'] 
                                    == 'pmid'):
#                                        hasPMID = True
#                                        pmidIndex = m
                                        tempData = {'ID': [refs[ref]['citation'][m]['pub-id']['#text']],
                                                   'IDtype':
                                                       [refs[ref]['citation'][m]['pub-id']['@pub-id-type']]}
                                        tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
#                            if hasPMID:
#                                tempData = {'ID': [refs[ref]['citation'][pmidIndex]['pub-id']['#text']],
#                                                   'IDtype':[refs[ref]['citation'][pmidIndex]['pub-id']['@pub-id-type']]}
#                                tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
#                            if hasDOI and not hasPMID:
#                                tempData = {'ID': [getPMIDfromDOI(refs[ref]['citation'][doiIndex]['pub-id']['#text'])],
#                                                       'IDtype': ['PMID']}
#                                tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                        refIDs = refIDs.append(tempDF, ignore_index=True)    
                if (i == 'element-citation'):
                    for l in refs[ref]['element-citation'].keys():
                        if (l == 'pub-id'):
                            if isinstance(refs[ref]['element-citation']['pub-id'], dict):
                                tempData = {'ID': [refs[ref]['element-citation']['pub-id']['#text']],
                                           'IDtype':
                                               [refs[ref]['element-citation']['pub-id']['@pub-id-type']]}
                                tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                            if isinstance(refs[ref]['element-citation']['pub-id'], list):
                                for index in range(0, len(refs[ref]['element-citation']['pub-id'])):
                                    if (refs[ref]['element-citation']['pub-id'][index]['@pub-id-type'] == 'pmid'):
                                        hasPMID = True
                                        pmidIndex = index
                                    if (refs[ref]['element-citation']['pub-id'][index]['@pub-id-type'] == 'doi'):
                                        hasDOI = True
                                        doiIndex = index
                                    if hasPMID:
                                        tempData = {'ID': [refs[ref]['element-citation']['pub-id'][pmidIndex]['#text']],
                                           'IDtype': [refs[ref]['element-citation']['pub-id'][pmidIndex]['@pub-id-type']]}
                                        tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                                        #refIDs = refIDs.append(tempDF, ignore_index=True)
                                    if hasDOI and not hasPMID:
                                        tempData = {'ID': [getPMIDfromDOI(refs[ref]['element-citation']['pub-id'][doiIndex]['#text'])],
                                                           'IDtype': ['PMID']}
                                        tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                            refIDs = refIDs.append(tempDF, ignore_index=True)
                if (i == 'mixed-citation'):
                    if (isinstance(refs[ref]['mixed-citation'], dict)):
                        for key in refs[ref]['mixed-citation'].keys():
                            if (key == 'pub-id'):
                                if (isinstance(refs[ref]['mixed-citation']['pub-id'], list)):
                                    for index in range(0, len(refs[ref]['mixed-citation']['pub-id'])):
                                        if (refs[ref]['mixed-citation']['pub-id'][index]['@pub-id-type'] == 'pmid'):
                                            hasPMID = True
                                            pmidIndex = index
                                        if (refs[ref]['mixed-citation']['pub-id'][index]['@pub-id-type'] == 'doi'):
                                            hasDOI = True
                                            doiIndex = index
                                    if hasPMID:
                                        tempData = {'ID': [refs[ref]['mixed-citation']['pub-id'][pmidIndex]['#text']],
                                           'IDtype': [refs[ref]['mixed-citation']['pub-id'][pmidIndex]['@pub-id-type']]}
                                        tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                                        #refIDs = refIDs.append(tempDF, ignore_index=True)
                                    if hasDOI and not hasPMID:
                                        tempData = {'ID': [getPMIDfromDOI(refs[ref]['mixed-citation']['pub-id'][doiIndex]['#text'])],
                                                           'IDtype': ['PMID']}
                                        tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                                if (isinstance(refs[ref]['mixed-citation']['pub-id'], dict)):
                                    tempData = {'ID': [refs[ref]['mixed-citation']['pub-id']['#text']],
                                           'IDtype':
                                               [refs[ref]['mixed-citation']['pub-id']['@pub-id-type']]}
                                    tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                                refIDs = refIDs.append(tempDF, ignore_index=True)
    if isinstance(refs, dict):
        pass
        # do stuff here
    
    refIDs = refIDs.loc[:, 'ID']
    
    tempDF2 = pd.DataFrame({'ID': [ID]}, columns=['ID'])
    tempDF2 = tempDF2.append(refIDs, ignore_index=True)
    tempDF2 = tempDF2.stack()
    
    mti_refs_short = mti_refs_short.append(tempDF2, ignore_index=True)
    
    # This is a delay in accordance with PubMed API usage guidelines.
    # It should not be set lower than .34.
    #time.sleep(.5)

# cleanup tasks:
# need to export error logs
