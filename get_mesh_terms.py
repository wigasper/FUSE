# sandbox for prototyping getting of MeSH terms
# how do I deal with qualifiers?

import os

from Bio import Entrez
from bs4 import BeautifulSoup
from tqdm import tqdm

# OSX path:
#os.chdir('/Users/wigasper/Documents/Research Project')

# Ubuntu path:
#os.chdir('/home/wkg/Documents/Research Project')
os.chdir('/media/wkg/storage/Research Project')

#ID = "10073946"
#Entrez.email = "kgasper@unomaha.edu"
#handle = Entrez.efetch(db="pubmed", id=ID, retmode="xml")
#handle = open("./PMC XMLs/{}.xml".format(ID), "r")

edge_list = pd.read_csv("./FUSE/edge_list.csv")
edge_list = edge_list.loc[:, ~edge_list.columns.str.contains("^Unnamed")]

ids_to_get = edge_list["1"].tolist()

# Drop duplicates:
ids_to_get = list(dict.fromkeys(ids_to_get))

#for ID in tqdm(edge_list[]):
    try:
        handle = open("./MeSH XMLs/{}.xml".format(ID), "r")
        soup = BeautifulSoup(handle.read())
        
        mesh_terms = [ID]
        
        #for mesh_heading in soup.find_all("meshheading"):
        #    if mesh_heading.descriptorname != None:
        #        #term_id = mesh_heading.descriptorname['ui']
        #        term_id = mesh_heading.descriptorname.string
        #        if mesh_heading.qualifiername != None:
        #            for qualifier in mesh_heading.find_all("qualifiername"):
        #                #term_id = "".join([term_id, ",", qualifier['ui']])
        #                term_id = "".join([term_id, "/", qualifier.string])
        #    mesh_terms.append(term_id)
            
        
        for mesh_heading in soup.find_all("meshheading"):
            if mesh_heading.descriptorname is not None:
                term_id = mesh_heading.descriptorname['ui']
                #term_id = mesh_heading.descriptorname.string
                ##### For dealing with qualifiers:
        #        if mesh_heading.qualifiername != None:
        #            for qualifier in mesh_heading.find_all("qualifiername"):
        #                #term_id = "".join([term_id, ",", qualifier['ui']])
        #                #term_id = "".join([term_id, "/", qualifier.string])
        #                mesh_terms.append("".join([term_id, "/", qualifier.string]))
        #        else:
        #            mesh_terms.append(term_id)
                mesh_terms.append(term_id)
            #mesh_terms.append(term_id)
            
print(mesh_terms)