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
#os.chdir('/home/wkg/Documents/Research Project')
os.chdir('/media/wkg/storage/Research Project')

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
            # was 148228/30806
# FNFE: ['168/14/1522', '129/10/1041', '166/8/828', '347/14/1087', '14/4/255', '16/Suppl_1/i9', '16/Suppl_1/i3', '39/7_suppl/22', '39/2/128', '165/9/986', '59/3/297', '162/12/2393', '18/1/24', '26/5/759', '347/16/1233', '64/2/169', '26/3/741', '27/1/34', '31/5/703', '147/4/505', '32/1/1', '8/1/7', '17/8/1006', '17/3/224', '289/18/2400', '165/22/2618', '19/5/287', '189/3/273', '17/2/148', '47/1/143', '17/2/159', '8/1/39', '355/15/1563', '15/12/3091', '30/4/737', '12/9/1426', '17/5/373', '145/11/1230', '351/1/13', '360/13/1278', '330/6005/759']
# PME: ['21179389', '17853104', '2110431087005', '015008', '21554748', '15975863', '328602', 'e14200', 'S0091674996003582', 'S0749379798001081', 'S009167490300006X', 'PMC3153544', '1524839909334621', '1524839906289983', 'S0740547202002301', '770296512', '788737774', '914290040', 'S0306460302002277', '1524839908324778', 'S0306460302002502', '1403494810393557', 'S0306460303000777', 'S0197245603000692', 'S0933365704000478', 'W525862811208880', '913643864', '926943837', '930981489', '930958227', '911056370', '17589070', 'S0895435698001310', 'S0140673600043373', '908748225', '912392848', '779266704', 'S009174350090731X', '1524839908330745', '20162763', '22832655', '14725281', '15975883', '19253042']