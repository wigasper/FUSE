import json

import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    uids = []
    with open("../data/mesh_data.tab", "r") as handle:
        for line in handle:
            line = line.strip("\n").split("\t")
            uids.append(line[0])

    with open("../data/term_freqs_rev_3_all_terms.json", "r") as handle:
        data = json.load(handle)

    # short one to prototype keras generator
    docs_list = list(data.keys())
    partition = int(len(docs_list) * .8)

    train_docs = docs_list[:partition]
    test_docs = docs_list[partition:]

    docs_list = set(docs_list)

    solution = {}
    with open("../data/pm_doc_term_counts.csv", "r") as handle:
        for line in handle:
            line = line.strip("\n").split(",")
            if line[0] in docs_list:
                solution[line[0]] = line[1:]

    with open("train_ids", "w") as out:
        for doc in tqdm(train_docs):
            out.write(f"{doc}\n")

    with open("test_ids", "w") as out:
        for doc in tqdm(test_docs):
            out.write(f"{doc}\n")

    for doc in tqdm(docs_list):
        row = []
        for uid in uids:
            if uid in data[doc].keys():
                # truncate to save space
                row.append(float(str(data[doc][uid])[:6]))
            else:
                row.append(0)
        row = np.array(row)
        np.save(f"data/{doc}_x.npy", row)
        
        row = []
        for uid in uids:
            if uid in solution[doc]:
                row.append(1)
            else:
                row.append(0)
        np.save(f"data/{doc}_y.npy", row)

