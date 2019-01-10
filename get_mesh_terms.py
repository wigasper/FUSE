# sandbox for prototyping getting of MeSH terms
# how do I deal with qualifiers?

import os

from Bio import Entrez
from bs4 import BeautifulSoup

# OSX path:
#os.chdir('/Users/wigasper/Documents/Research Project')

# Ubuntu path:
#os.chdir('/home/wkg/Documents/Research Project')
os.chdir('/media/wkg/storage/Research Project')

ID = "10073946"
Entrez.email = "kgasper@unomaha.edu"
handle = Entrez.efetch(db="pubmed", id=ID, retmode="xml")
#handle = open("./PMC XMLs/{}.xml".format(ID), "r")

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
    if mesh_heading.descriptorname != None:
        #term_id = mesh_heading.descriptorname['ui']
        term_id = mesh_heading.descriptorname.string
        if mesh_heading.qualifiername != None:
            for qualifier in mesh_heading.find_all("qualifiername"):
                #term_id = "".join([term_id, ",", qualifier['ui']])
                #term_id = "".join([term_id, "/", qualifier.string])
                mesh_terms.append("".join([term_id, "/", qualifier.string]))
        else:
            mesh_terms.append(term_id)
    #mesh_terms.append(term_id)
    
print(mesh_terms)