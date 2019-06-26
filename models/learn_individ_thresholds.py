#!/usr/bin/env python3
import json
import time

def main():
    # Load in term frequencies
    with open("../data/term_freqs.json", "r") as handle:
        temp = json.load(handle)
    # original file clearly needs to be rebuilt as dict
    term_freqs = {}
    for doc in temp:
        term_freqs[doc[0]] = doc[1]

    # Load in solution values
    with open("../data/baseline_solution.json", "r") as handle:
        solution = json.load(handle)

    # Load in UIDs
    uids = []
    with open("../data/mesh_data.tab", "r") as handle:
        for line in handle:
            line = line.strip("\n").split("\t")
            uids.append(line[0])

    # Get max freq for docs with more than 10 terms applied to their citations
    max_freq = 0
    for doc in term_freqs:
        for term in term_freqs[doc].keys():
            if len(term_freqs[doc]) > 10 and term_freqs[doc][term] > max_freq:
                max_freq = term_freqs[doc][term]

    # Divide values by max
    for doc in term_freqs:
        for term in term_freqs[doc].keys():
            term_freqs[doc][term] = term_freqs[doc][term] / max_freq
            
    # Dict to store the optimal thresholds for each UID
    uid_thresholds = {uid: 0 for uid in uids}

    thresholds = [x * .005 for x in range(0,200)]

    for uid in uids:
        precisions = []
        recalls = []
        f1s = []
        for thresh in thresholds:
            predictions = {doc: 0 for doc in term_freqs.keys()}
            
            for doc in term_freqs.keys():
                if uid in term_freqs[doc].keys() and term_freqs[doc][uid] > thresh:
                    predictions[doc] = 1

            true_pos = 0
            false_pos = 0
            false_neg = 0
            
            for pmid in predictions.keys():
                if predictions[pmid] == 1 and uid in solution[pmid]:
                    true_pos += 1
                if predictions[pmid] == 1 and uid not in solution[pmid]:
                    false_pos += 1
                if predictions[pmid] == 0 and uid in solution[pmid]:
                    false_neg += 1
            
            if true_pos == 0:
                precision = 0
                recall = 0
                f1 = 0
            else:
                precision = true_pos / (true_pos + false_pos)
                recall = true_pos / (true_pos + false_neg)
                f1 = (2 * precision * recall) / (precision + recall)
        
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        max_f1 = max(f1s)
        max_thresh = thresholds[f1s.index(max_f1)]
        print("UID: " + uid + " - Max f1: " + str(max_f1) + " - at thresh: " + str(max_thresh))

        uid_thresholds[uid] = max_thresh

    with open ("../data/individual_term_thresholds.json", "w") as out:
        json.dump(uid_thresholds, out)

if __name__ == "__main__":
    main()