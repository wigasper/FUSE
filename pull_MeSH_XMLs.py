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

edge_list = pd.read_csv("edge_list.csv", index_col=None)
edge_list = edge_list.replace(" 10.1007/s11606-011-1968-2", "22282311")
edge_list = edge_list.replace("120/4/e902", "17908746")
edge_list = edge_list.replace("121/3/575", "18310208")
edge_list = edge_list.replace("353/5/487", "16079372")
edge_list = edge_list.replace("163/2/141", "19188646")
edge_list = edge_list.replace("13/7/930", "18809644")


ids_to_get = edge_list['1'].tolist()
del(edge_list)

# Drop duplicates:
ids_to_get = list(dict.fromkeys(ids_to_get))

# pm_errors stores any errors from PubMed's side
pm_errors = []
fnfe= []

for ID in tqdm(ids_to_get):
    try:
        start_time = time.perf_counter()
        file = Path("./MeSH XMLs/{}.xml".format(ID))
    
        if not file.exists():
            Entrez.email = "kgasper@unomaha.edu"
            handle = Entrez.efetch(db="pubmed", id=ID, retmode="xml")
            xmlString = handle.read()
            element = xmltodict.parse(xmlString)
        
            pm_error = False
        
            # Check for an error on PMC's side and record it
            if isinstance(element['PubmedArticleSet'], dict):
                for key in element['PubmedArticleSet'].keys():
                    if key == 'error':
                        pm_errors.append(ID)
                        pm_error = True
                if not pm_error:
                    file_out = open("./MeSH XMLs/{}.xml".format(ID), "w")
                    file_out.write(xmlString)
            if not isinstance(element['PubmedArticleSet'], dict):
                pm_errors.append(ID)
            # This is a delay in accordance with PubMed API usage guidelines.
            if time.perf_counter() - start_time < .4:
                time.sleep(.4 - (time.perf_counter() - start_time))
    except FileNotFoundError:
        fnfe.append(ID)
        
#start = time.perf_counter()
#time.sleep(.3)
#time.perf_counter() - start
#
#for i in tqdm(range(100)):
#    start = time.perf_counter()
#    if time.perf_counter() - start < .33:
#        time.sleep(.33 - (time.perf_counter() - start))
            
####################
# Errors: 21179389
# FileNotFoundError: [Errno 2] No such file or directory: './MeSH XMLs/ 10.1007/s11606-011-1968-2.xml'
            # was 148228/308067