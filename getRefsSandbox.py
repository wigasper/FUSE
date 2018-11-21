#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 12:21:45 2018

@author: wigasper
"""
##########
#########
# need to look at PMC2707857. different structure.
import xmltodict
from Bio import Entrez
import pandas as pd
import time
import os

#### random testing area
#### okay I can use this search function to retrive a PMID from a doi
handle = Entrez.esearch(db="pubmed", term="10.1186/1471-2377-6-33")
record1 = Entrez.read(handle)

#######

os.chdir('/Users/wigasper/Documents/Research Project')

mti_oaSubset_train = pd.read_csv("2013_MTI_in_OA_train.csv")

# This top section is a testing area for pulling single articles
# from PMC and manipulating them.

randomPMID = "PMC2707857"
Entrez.email = "kgasper@unomaha.edu"
handle = Entrez.efetch(db="pmc", id=randomPMID, retmode="xml")
xmlString = handle.read()

element = xmltodict.parse(xmlString)

# This boolean value describes whether or not a reference list exists in 
# an article. With the current logic it is needed to prevent KeyErrors.
hasRefList = False

# Check for a reference list, assign it, and set hasRefList true.
for i in element['pmc-articleset']['article']['back'].keys():
    if (i == 'ref-list'):
        refs = element['pmc-articleset']['article']['back']['ref-list']['ref']
        hasRefList = True

refIDs = pd.DataFrame(columns=['ID', 'IDtype'])

# This gets all the references for the single PMCID and adds them to a
# dataframe, refIDs
for i in range(0, len(refs)):
    for j in refs[i].keys():
        if (j == 'citation' and hasRefList):
            for k in refs[i]['citation'].keys():
                if (k == 'pub-id'):
                    tempData = {'ID': [refs[i]['citation']['pub-id']['#text']],
                                'IDtype':
                                    [refs[i]['citation']['pub-id']['@pub-id-type']]}
                    tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                    refIDs = refIDs.append(tempDF, ignore_index=True)
        if (j == 'element-citation' and hasRefList):
            for l in refs[i]['element-citation'].keys():
                if (l == 'pub-id'):
                    tempData = {'ID': [refs[i]['element-citation']['pub-id']['#text']],
                                'IDtype':
                                    [refs[i]['element-citation']['pub-id']['@pub-id-type']]}
                    tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                    refIDs = refIDs.append(tempDF, ignore_index=True)

refIDs = refIDs.loc[:, 'ID']

#test = pd.DataFrame({'ID': [123456]}, columns=['ID'])
#test = test.append(refIDs, ignore_index=True)
#test = test.stack()

# This is the testing area for pulling the references for multiple articles
# and putting them all into a data frame together.
mti_oa_short = mti_oaSubset_train[0:10]
mti_refs_short = pd.DataFrame()

for ID in mti_oa_short['Accession ID']:
    Entrez.email = "kgasper@unomaha.edu"
    handle = Entrez.efetch(db="pmc", id=ID, retmode="xml")
    xmlString = handle.read()
    
    element = xmltodict.parse(xmlString)
    
    # These boolean values describe whether or XML tags exist in 
    # an article. With the current logic it is needed to prevent KeyErrors.
    hasRefList = False
    hasCitation = False
    hasElementCitation = False
    
    # Check for a reference list, assign it, and set hasRefList true.
    for i in element['pmc-articleset']['article']['back'].keys():
        if (i == 'ref-list'):
            refs = element['pmc-articleset']['article']['back']['ref-list']['ref']
            hasRefList = True
    
    if (hasRefList):
        for i in refs[0].keys():
            if (i == 'citation'):
                hasCitation = True
                
    if (hasRefList):
        for i in refs[0].keys():
            if (i == 'element-citation'):
                hasElementCitation = True

    # This is a temporary data frame used to hold all the reference IDs for
    # one article. IDtype may be unnecessary.
    refIDs = pd.DataFrame(columns=['ID', 'IDtype'])

    for ref in range(0, len(refs)):
        if (hasCitation):
            for k in refs[ref]['citation'].keys():
                if (k == 'pub-id'):
                    tempData = {'ID': [refs[ref]['citation']['pub-id']['#text']],
                               'IDtype':
                                   [refs[ref]['citation']['pub-id']['@pub-id-type']]}
                    tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                    refIDs = refIDs.append(tempDF, ignore_index=True)
        if (hasElementCitation):
            for l in refs[ref]['element-citation'].keys():
                if (l == 'pub-id'):
                    tempData = {'ID': [refs[ref]['element-citation']['pub-id']['#text']],
                               'IDtype':
                                   [refs[ref]['element-citation']['pub-id']['@pub-id-type']]}
                    tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                    refIDs = refIDs.append(tempDF, ignore_index=True)

    refIDs = refIDs.loc[:, 'ID']
    
    tempDF2 = pd.DataFrame({'ID': [ID]}, columns=['ID'])
    tempDF2 = tempDF2.append(refIDs, ignore_index=True)
    tempDF2 = tempDF2.stack()
    
    mti_refs_short = mti_refs_short.append(tempDF2, ignore_index=True)
    
    # This is a delay in accordance with PubMed API usage guidelines.
    # It should not be set lower than .34.
    time.sleep(.5)
    
###########################################################
##### new trial to try and check if citation key is a list
#### problems here
mti_oa_short = mti_oaSubset_train[0:80]
mti_refs_short = pd.DataFrame()

for ID in mti_oa_short['Accession ID']:
    Entrez.email = "kgasper@unomaha.edu"
    handle = Entrez.efetch(db="pmc", id=ID, retmode="xml")
    xmlString = handle.read()
    
    element = xmltodict.parse(xmlString)
    
    # These boolean values describe whether or XML tags exist in 
    # an article. With the current logic it is needed to prevent KeyErrors.
    hasRefList = False
    hasCitationDict = False
    hasElementCitation = False
    hasPubID = False
    hasCitationList = False
    
    # Check for a reference list, assign it, and set hasRefList true.
    for i in element['pmc-articleset']['article']['back'].keys():
        if (i == 'ref-list'):
            refs = element['pmc-articleset']['article']['back']['ref-list']['ref']
            hasRefList = True
    
    if (hasRefList):
        for i in refs[0].keys():
            if (i == 'citation'):
                #if (type(refs[0]['citation']) is list):
                if (isinstance(refs[0]['citation'], list)):
                    hasCitationList = True
    
    if (hasRefList):
        for i in refs[0].keys():
            if (i == 'citation'):
                if (isinstance(refs[0]['citation'], dict)):
                    hasCitationDict = True
        if (hasCitationDict):
            for j in refs[0]['citation'].keys():
                if (j == 'pub-id'):
                    hasPubID = True
                
    if (hasRefList):
        for i in refs[0].keys():
            if (i == 'element-citation'):
                hasElementCitation = True
        if (hasElementCitation):
            for j in refs[0]['element-citation'].keys():
                if (j == 'pub-id'):
                    hasPubID = True    

    # This is a temporary data frame used to hold all the reference IDs for
    # one article. IDtype may be unnecessary.
    # there is a keyError here. also current logic screws it all up. may have missed something
    refIDs = pd.DataFrame(columns=['ID', 'IDtype'])

    for ref in range(0, len(refs)):
        if (hasCitationDict):
            for k in refs[ref]['citation'].keys():
                if (k == 'pub-id'):
                    tempData = {'ID': [refs[ref]['citation']['pub-id']['#text']],
                               'IDtype':
                                   [refs[ref]['citation']['pub-id']['@pub-id-type']]}
                    tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                    refIDs = refIDs.append(tempDF, ignore_index=True)

        if (hasElementCitation):
            for l in refs[ref]['element-citation'].keys():
                if (l == 'pub-id'):
                    tempData = {'ID': [refs[ref]['element-citation']['pub-id']['#text']],
                               'IDtype':
                                   [refs[ref]['element-citation']['pub-id']['@pub-id-type']]}
                    tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                    refIDs = refIDs.append(tempDF, ignore_index=True)

        if (hasCitationList):
            for m in range(0, len(refs[ref]['citation'])):
                for n in refs[ref]['citation'][m].keys():
                    if (n == 'pub-id'):
                        if (refs[ref]['citation'][m]['pub-id']['@pub-id-type'] 
                        == 'doi'):
                            print("oh it happened")
                        if (refs[ref]['citation'][m]['pub-id']['@pub-id-type'] 
                        == 'pmid'):
                            tempData = {'ID': [refs[ref]['citation'][m]['pub-id']['#text']],
                               'IDtype':
                                   [refs[ref]['citation'][m]['pub-id']['@pub-id-type']]}
                            tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                    refIDs = refIDs.append(tempDF, ignore_index=True)
    refIDs = refIDs.loc[:, 'ID']
    
    tempDF2 = pd.DataFrame({'ID': [ID]}, columns=['ID'])
    tempDF2 = tempDF2.append(refIDs, ignore_index=True)
    tempDF2 = tempDF2.stack()
    
    mti_refs_short = mti_refs_short.append(tempDF2, ignore_index=True)
    
    # This is a delay in accordance with PubMed API usage guidelines.
    # It should not be set lower than .34.
    time.sleep(.5)


###### Mostly same stuff as before. need to troubleshoot this.
###### could reduce the nesting and make everything more nicely readable
# This is the testing area for pulling the references for multiple articles
# and putting them all into a data frame together.
mti_oa_short = mti_oaSubset_train[0:10]
mti_refs_short = pd.DataFrame()

for ID in mti_oa_short['Accession ID']:
    Entrez.email = "kgasper@unomaha.edu"
    handle = Entrez.efetch(db="pmc", id=ID, retmode="xml")
    xmlString = handle.read()
    
    element = xmltodict.parse(xmlString)
    
    # These boolean values describe whether or XML tags exist in 
    # an article. With the current logic it is needed to prevent KeyErrors.
    hasRefList = False
    hasCitation = False
    hasElementCitation = False
    hasPubID = False
    
    # Check for a reference list, assign it, and set hasRefList true.
    for i in element['pmc-articleset']['article']['back'].keys():
        if (i == 'ref-list'):
            refs = element['pmc-articleset']['article']['back']['ref-list']['ref']
            hasRefList = True
    
    if (hasRefList):
        for i in refs[0].keys():
            if (i == 'citation'):
                hasCitation = True
        if (hasCitation):
            for j in refs[0]['citation'].keys():
                if (j == 'pub-id'):
                    hasPubID = True
                
    if (hasRefList):
        for i in refs[0].keys():
            if (i == 'element-citation'):
                hasElementCitation = True
        if (hasElementCitation):
            for j in refs[0]['element-citation'].keys():
                if (j == 'pub-id'):
                    hasPubID = True

    # This is a temporary data frame used to hold all the reference IDs for
    # one article. IDtype may be unnecessary.
    refIDs = pd.DataFrame(columns=['ID', 'IDtype'])

    for ref in range(0, len(refs)):
        if (hasCitation and hasPubID):
            tempData = {'ID': [refs[ref]['citation']['pub-id']['#text']],
                               'IDtype': [refs[ref]['citation']['pub-id']['@pub-id-type']]}
            tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
            refIDs = refIDs.append(tempDF, ignore_index=True)
        if (hasElementCitation and hasPubID):
            tempData = {'ID': [refs[ref]['element-citation']['pub-id']['#text']],
                               'IDtype':
                                   [refs[ref]['element-citation']['pub-id']['@pub-id-type']]}
            tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
            refIDs = refIDs.append(tempDF, ignore_index=True)

    refIDs = refIDs.loc[:, 'ID']
    
    tempDF2 = pd.DataFrame({'ID': [ID]}, columns=['ID'])
    tempDF2 = tempDF2.append(refIDs, ignore_index=True)
    tempDF2 = tempDF2.stack()
    
    mti_refs_short = mti_refs_short.append(tempDF2, ignore_index=True)
    
    # This is a delay in accordance with PubMed API usage guidelines.
    # It should not be set lower than .34.
    time.sleep(1)