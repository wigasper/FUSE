#!/usr/bin/env python3

from bs4 import BeautifulSoup


records = []
with open("./data/desc2019", "r") as handle:
    record = ""
    for line in handle:
        if line.startswith("<DescriptorRecord"):
            if record:
                records.append(record)
                record = ""
            record = "".join([record, line])
        else:
            record = "".join([record, line])

# Discard first 2 records which has document info
records = records[2:]

desc_records = []
desc_uis = []
desc_names = []
tree_num_lists = []

for rec in records:
    soup = BeautifulSoup(rec)
    #soup.find_all('descriptorui')

    #for record in soup.find_all('descriptorrecord'):
        #descriptoruis = []
        #desc_records.append(record)
    desc_uis.append(soup.descriptorui.string)
    desc_names.append(soup.descriptorname.find('string').string)
    tree_nums = []
    if soup.treenumberlist is not None:
        for tree_num in soup.treenumberlist.find_all('treenumber'):
            tree_nums.append(tree_num.string)
            
    tree_num_lists.append(tree_nums)
        #tree_nums.append(record.treenumberlist.treenumber.string)
        #for desc in record.find_all('descriptorui'):
        #    descriptoruis.append(desc)