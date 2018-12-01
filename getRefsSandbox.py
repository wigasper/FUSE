# Author: Kirk Gasper

import xmltodict
from Bio import Entrez
import pandas as pd
import time
import os
from tqdm import tqdm

os.chdir('/Users/wigasper/Documents/Research Project')

mti_oaSubset_train = pd.read_csv("2013_MTI_in_OA_train.csv")

# This method takes a DOI ID and returns a PMID if Pubmed has
# one for the DOI ID; if not, returns False
def getPMIDfromDOI(doiIn):
    time.sleep(.34)
    Entrez.email = "kgasper@unomaha.edu"
    handle = Entrez.esearch(db="pubmed", term=doiIn)
    record = Entrez.read(handle)
    if len(record['IdList']) > 0:
        PMID = record['IdList'][0]
        return PMID
    else:
        return False

# This top section is used to pull single articles for examination

#randomPMID = "PMC3305422"
#Entrez.email = "kgasper@unomaha.edu"
#handle = Entrez.efetch(db="pmc", id=randomPMID, retmode="xml")
#xmlString = handle.read()
#
#element = xmltodict.parse(xmlString)

############################################
# This is the testing area for pulling the references for multiple articles
# and putting them all into a data frame together.

# TO DO:
# need to do logging with python's logger, not these DFs
# PMC3412080 has both ref-listref-list and ref-list> refs

mti_oa_short = mti_oaSubset_train
mti_refs_short = pd.DataFrame()

for ID in tqdm(mti_oa_short['Accession ID']):
#    Entrez.email = "kgasper@unomaha.edu"
#    handle = Entrez.efetch(db="pmc", id=ID, retmode="xml")
    handle = open("./PMC XMLs/{}.xml".format(ID), "r")
    xmlString = handle.read()
    
    element = xmltodict.parse(xmlString)
    
    # These boolean values describe whether or XML tags exist in 
    # an article. With the current logic it is needed to prevent KeyErrors.

    has_ref_list = False
    hasElementCitation = False
    hasCitation = False
    hasPMID = False
    hasDOI = False
    has_pubID_dict_pmid = False
    has_pubID_dict_doi = False
    
    # Check for a reference list, assign it, and set hasRefList true.
    for j in element['pmc-articleset']['article'].keys():
        if j == 'back':
            for i in element['pmc-articleset']['article']['back'].keys():
                if i == 'ref-list':
                    for k in element['pmc-articleset']['article']['back']['ref-list'].keys():
                        if k == 'ref':
                            refs = element['pmc-articleset']['article']['back']['ref-list']['ref']
                            has_ref_list = True
                        if k == 'ref-list':
                            refs = element['pmc-articleset']['article']['back']['ref-list']['ref-list']['ref']
                            has_ref_list = True
    # TO DO:
    # need to put some logging here if an article doesn't have 'ref-list'
    # record which articles don't have it for examination

    # This is a temporary data frame used to hold all the reference IDs for
    # one article. IDtype may be unnecessary.
    refIDs = pd.DataFrame(columns=['ID', 'IDtype'])
    
    # This mess extracts the references from PMC article XML files
    if isinstance(refs, list) and has_ref_list:
        for ref in range(0, len(refs)):
            for i in refs[ref].keys():
                if i == 'citation':
                    if isinstance(refs[ref]['citation'], dict):
                        for k in refs[ref]['citation'].keys():
                            if k == 'pub-id':
                                if refs[ref]['citation']['pub-id']['@pub-id-type'] == 'pmid':
                                    tempData = {'ID': [refs[ref]['citation']['pub-id']['#text']],
                                               'IDtype':
                                                   [refs[ref]['citation']['pub-id']['@pub-id-type']]}
                                    tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                                if refs[ref]['citation']['pub-id']['@pub-id-type'] == 'doi':
                                    tempData = {'ID': [getPMIDfromDOI(refs[ref]['citation']['pub-id']['#text'])],
                                                                       'IDtype': ['PMID']}
                                    tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                                refIDs = refIDs.append(tempDF, ignore_index=True)
                    if isinstance(refs[ref]['citation'], list):
                        for m in range(0, len(refs[ref]['citation'])):
                            for n in refs[ref]['citation'][m].keys():
                                if n == 'pub-id':
                                    if isinstance(refs[ref]['citation'][m]['pub-id'], dict):
                                        if refs[ref]['citation'][m]['pub-id']['@pub-id-type'] == 'pmid':
                                            has_pubID_dict_pmid = True
                                            cit_list_index_for_pmid = m
                                        if refs[ref]['citation'][m]['pub-id']['@pub-id-type'] == 'doi':
                                            has_pubID_dict_doi = True
                                            cit_list_index_for_doi = m
                                    if isinstance(refs[ref]['citation'][m]['pub-id'], list):
                                        for index in range(0, len(refs[ref]['citation'][m]['pub-id'])):
                                            if refs[ref]['citation'][m]['pub-id'][index]['@pub-id-type'] == 'pmid':
                                                hasPMID = True
                                                pmidIndex = index
                                            if refs[ref]['citation'][m]['pub-id'][index]['@pub-id-type'] == 'doi':
                                                hasDOI = True
                                                doiIndex = index
                                            if hasPMID:
                                                tempData = {'ID': [refs[ref]['citation'][m]['pub-id'][pmidIndex]['#text']],
                                                   'IDtype': [refs[ref]['citation'][m]['pub-id'][pmidIndex]['@pub-id-type']]}
                                                tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                                            if hasDOI and not hasPMID:
                                                tempData = {'ID': [getPMIDfromDOI(refs[ref]['citation']['pub-id'][doiIndex]['#text'])],
                                                                   'IDtype': ['PMID']}
                                                tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                                            hasDOI = False
                                            hasPMID = False
                                        refIDs = refIDs.append(tempDF, ignore_index=True)
                            if has_pubID_dict_pmid:
                                tempData = {'ID': [refs[ref]['citation'][cit_list_index_for_pmid]['pub-id']['#text']],
                                               'IDtype':
                                                   [refs[ref]['citation'][cit_list_index_for_pmid]['pub-id']['@pub-id-type']]}
                                tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                                refIDs = refIDs.append(tempDF, ignore_index=True)

                            if has_pubID_dict_doi and not has_pubID_dict_pmid:
                                tempData = {'ID': [getPMIDfromDOI(refs[ref]['citation'][cit_list_index_for_doi]['pub-id']['#text'])],
                                                                   'IDtype': ['PMID']}
                                tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                                refIDs = refIDs.append(tempDF, ignore_index=True)
                            has_pubID_dict_pmid = False
                            has_pubID_dict_doi = False
                if i == 'element-citation':
                    if isinstance(refs[ref]['element-citation'], dict):
                        for l in refs[ref]['element-citation'].keys():
                            if l == 'pub-id':
                                if isinstance(refs[ref]['element-citation']['pub-id'], dict):
                                    tempData = {'ID': [refs[ref]['element-citation']['pub-id']['#text']],
                                               'IDtype':
                                                   [refs[ref]['element-citation']['pub-id']['@pub-id-type']]}
                                    tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                                if isinstance(refs[ref]['element-citation']['pub-id'], list):
                                    for index in range(0, len(refs[ref]['element-citation']['pub-id'])):
                                        if refs[ref]['element-citation']['pub-id'][index]['@pub-id-type'] == 'pmid':
                                            hasPMID = True
                                            pmidIndex = index
                                        if refs[ref]['element-citation']['pub-id'][index]['@pub-id-type'] == 'doi':
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
                                        hasPMID = False
                                        hasDOI = False
                                refIDs = refIDs.append(tempDF, ignore_index=True)
                    if isinstance(refs[ref]['element-citation'], list):
                        for m in range(0, len(refs[ref]['element-citation'])):
                                for n in refs[ref]['element-citation'][m].keys():
                                    if n == 'pub-id':
                                        if isinstance(refs[ref]['element-citation'][m]['pub-id'], dict):
                                            if refs[ref]['element-citation'][m]['pub-id']['@pub-id-type'] == 'pmid':
                                                has_pubID_dict_pmid = True
                                                cit_list_index_for_pmid = m
                                            if refs[ref]['element-citation'][m]['pub-id']['@pub-id-type'] == 'doi':
                                                has_pubID_dict_doi = True
                                                cit_list_index_for_doi = m
                                        if isinstance(refs[ref]['element-citation'][m]['pub-id'], list):
                                            for index in range(0, len(refs[ref]['element-citation'][m]['pub-id'])):
                                                if refs[ref]['element-citation'][m]['pub-id'][index]['@pub-id-type'] == 'pmid':
                                                    hasPMID = True
                                                    pmidIndex = index
                                                if refs[ref]['element-citation'][m]['pub-id'][index]['@pub-id-type'] == 'doi':
                                                    hasDOI = True
                                                    doiIndex = index
                                                if hasPMID:
                                                    tempData = {'ID': [refs[ref]['element-citation'][m]['pub-id'][pmidIndex]['#text']],
                                                       'IDtype': [refs[ref]['element-citation'][m]['pub-id'][pmidIndex]['@pub-id-type']]}
                                                    tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                                                if hasDOI and not hasPMID:
                                                    tempData = {'ID': [getPMIDfromDOI(refs[ref]['element-citation']['pub-id'][doiIndex]['#text'])],
                                                                       'IDtype': ['PMID']}
                                                    tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                                                hasDOI = False
                                                hasPMID = False
                                            refIDs = refIDs.append(tempDF, ignore_index=True)
                                if has_pubID_dict_pmid:
                                    tempData = {'ID': [refs[ref]['element-citation'][cit_list_index_for_pmid]['pub-id']['#text']],
                                                   'IDtype':
                                                       [refs[ref]['element-citation'][cit_list_index_for_pmid]['pub-id']['@pub-id-type']]}
                                    tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                                    refIDs = refIDs.append(tempDF, ignore_index=True)
    
                                if has_pubID_dict_doi and not has_pubID_dict_pmid:
                                    tempData = {'ID': [getPMIDfromDOI(refs[ref]['element-citation'][cit_list_index_for_doi]['pub-id']['#text'])],
                                                                       'IDtype': ['PMID']}
                                    tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                                    refIDs = refIDs.append(tempDF, ignore_index=True)
                                has_pubID_dict_pmid = False
                                has_pubID_dict_doi = False    
                if i == 'mixed-citation':
                    if isinstance(refs[ref]['mixed-citation'], dict):
                        for key in refs[ref]['mixed-citation'].keys():
                            if key == 'pub-id':
                                if isinstance(refs[ref]['mixed-citation']['pub-id'], list):
                                    for index in range(0, len(refs[ref]['mixed-citation']['pub-id'])):
                                        if refs[ref]['mixed-citation']['pub-id'][index]['@pub-id-type'] == 'pmid':
                                            hasPMID = True
                                            pmidIndex = index
                                        if refs[ref]['mixed-citation']['pub-id'][index]['@pub-id-type'] == 'doi':
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
                                    hasPMID = False
                                    hasDOI = False
                                if isinstance(refs[ref]['mixed-citation']['pub-id'], dict):
                                    tempData = {'ID': [refs[ref]['mixed-citation']['pub-id']['#text']],
                                           'IDtype':
                                               [refs[ref]['mixed-citation']['pub-id']['@pub-id-type']]}
                                    tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                                refIDs = refIDs.append(tempDF, ignore_index=True)
    if isinstance(refs, dict) and has_ref_list:
        for a in refs.keys():
            if a == 'mixed-citation':
                if isinstance(refs['mixed-citation'], dict):
                    for key in refs['mixed-citation'].keys():
                        if key == 'pub-id':
                            if isinstance(refs['mixed-citation']['pub-id'], list):
                                for index in range(0, len(refs['mixed-citation']['pub-id'])):
                                    if refs['mixed-citation']['pub-id'][index]['@pub-id-type'] == 'pmid':
                                        hasPMID = True
                                        pmidIndex = index
                                    if refs['mixed-citation']['pub-id'][index]['@pub-id-type'] == 'doi':
                                        hasDOI = True
                                        doiIndex = index
                                if hasPMID:
                                    tempData = {'ID': [refs['mixed-citation']['pub-id'][pmidIndex]['#text']],
                                       'IDtype': [refs['mixed-citation']['pub-id'][pmidIndex]['@pub-id-type']]}
                                    tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                                    #refIDs = refIDs.append(tempDF, ignore_index=True)
                                if hasDOI and not hasPMID:
                                    tempData = {'ID': [getPMIDfromDOI(refs['mixed-citation']['pub-id'][doiIndex]['#text'])],
                                                       'IDtype': ['PMID']}
                                    tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                                hasPMID = False
                                hasDOI = False
                            if isinstance(refs['mixed-citation']['pub-id'], dict):
                                tempData = {'ID': [refs['mixed-citation']['pub-id']['#text']],
                                       'IDtype':
                                           [refs['mixed-citation']['pub-id']['@pub-id-type']]}
                                tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                            refIDs = refIDs.append(tempDF, ignore_index=True)
            if a == 'element-citation':
                if isinstance(refs['element-citation'], dict):
                    for key in refs['element-citation'].keys():
                        if key == 'pub-id':
                            if isinstance(refs['element-citation']['pub-id'], list):
                                for index in range(0, len(refs['element-citation']['pub-id'])):
                                    if refs['element-citation']['pub-id'][index]['@pub-id-type'] == 'pmid':
                                        hasPMID = True
                                        pmidIndex = index
                                    if refs['element-citation']['pub-id'][index]['@pub-id-type'] == 'doi':
                                        hasDOI = True
                                        doiIndex = index
                                if hasPMID:
                                    tempData = {'ID': [refs['element-citation']['pub-id'][pmidIndex]['#text']],
                                       'IDtype': [refs['element-citation']['pub-id'][pmidIndex]['@pub-id-type']]}
                                    tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                                    #refIDs = refIDs.append(tempDF, ignore_index=True)
                                if hasDOI and not hasPMID:
                                    tempData = {'ID': [getPMIDfromDOI(refs['element-citation']['pub-id'][doiIndex]['#text'])],
                                                       'IDtype': ['PMID']}
                                    tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                                hasPMID = False
                                hasDOI = False
                            if isinstance(refs['element-citation']['pub-id'], dict):
                                tempData = {'ID': [refs['element-citation']['pub-id']['#text']],
                                       'IDtype':
                                           [refs['element-citation']['pub-id']['@pub-id-type']]}
                                tempDF = pd.DataFrame(tempData, columns=['ID', 'IDtype'])
                            refIDs = refIDs.append(tempDF, ignore_index=True)

    
    refIDs = refIDs.loc[:, 'ID']
    
    tempDF2 = pd.DataFrame({'ID': [ID]}, columns=['ID'])
    tempDF2 = tempDF2.append(refIDs, ignore_index=True)
    tempDF2 = tempDF2.stack()
    
    mti_refs_short = mti_refs_short.append(tempDF2, ignore_index=True)
    
# Export DF
mti_refs_short.to_csv('get_refs_run4.csv')
