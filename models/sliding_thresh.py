#!/usr/bin/env python3
import json
import logging

from tqdm import tqdm

# Default threshold function
def get_f1(thresh, term_freqs, solution):
    # Predict
    predictions = {}
    for doc in term_freqs.keys():
        predictions[doc] = [key for key, val in term_freqs[doc].items() if val > thresh]
        
    # Get evaluation metrics
    true_pos = 0
    false_pos = 0
    false_neg = 0
    
    for pmid in predictions:
        true_pos += len([pred for pred in predictions[pmid] if pred in solution[pmid]])
        false_pos += len([pred for pred in predictions[pmid] if pred not in solution[pmid]])
        false_neg += len([sol for sol in solution[pmid] if sol not in predictions[pmid]])

    if true_pos == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def learn_default_threshold(term_freqs, solution):
    curr_thresh = 0.0
    step_val = 0.001
    ##
    f1s = []
    
    f1s.append(get_f1(curr_thresh, term_freqs, solution))
    f1s.append(get_f1(curr_thresh + step_val, term_freqs, solution))
    
    curr_thresh += step_val
    next_thresh_f1 = get_f1(curr_thresh + step_val, term_freqs, solution)
    
    while not (next_thresh_f1 < f1s[-1] and next_thresh_f1 < f1s[-2] and f1s[-1] < f1s[-2]):
        curr_thresh += step_val
        f1s.append(get_f1(curr_thresh, term_freqs, solution))
        next_thresh_f1 = get_f1(curr_thresh + step_val, term_freqs, solution)
    
    print(curr_thresh - step_val)
    return curr_thresh - step_val

def predict(test_freqs, thresh, solution):

    # Test it out
    predictions = {}
            
    # Predict
    for doc in test_freqs.keys():
        mean_freq = sum(test_freqs[doc].values()) / len(test_freqs[doc])
        predictions[doc] = [key for key, val in test_freqs[doc].items() if val > (mean_freq + .009)]
        #if mean_freq < thresh:
        #    predictions[doc] = [key for key, val in test_freqs[doc].items() if val > thresh]
        #else:
        #    predictions[doc] = [key for key, val in test_freqs[doc].items() if val > mean_freq]
    #for doc in test_freqs.keys():
    #    predictions[doc] = [key for key, val in test_freqs[doc].items() if val > thresh]
    #for doc in test_freqs.keys():
    #    individ_thresh = sum(test_freqs[doc].values()) / len(test_freqs[doc])
    #    predictions[doc] = [key for key, val in test_freqs[doc].items() if val > individ_thresh]
        
    # Get evaluation metrics
    true_pos = 0
    false_pos = 0
    false_neg = 0
    
    for pmid in predictions:
        true_pos += len([pred for pred in predictions[pmid] if pred in solution[pmid]])
        false_pos += len([pred for pred in predictions[pmid] if pred not in solution[pmid]])
        false_neg += len([sol for sol in solution[pmid] if sol not in predictions[pmid]])

    if true_pos == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f1 = (2 * precision * recall) / (precision + recall)

    return predictions, f1

def train(train_freqs, solution, logger):
    return learn_default_threshold(train_freqs, solution)


def main():
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("../logs/sliding_thresh.log")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Load in term frequencies and partition
    with open("../data/term_freqs_rev_3_all_terms.json", "r") as handle:
        temp = json.load(handle)

    #temp = {key: val for key, val in temp_unfiltered.items() if 
    #        (sum(temp_unfiltered[key].values()) / len(temp_unfiltered[key])) > 0.016}
    
    docs_list = list(temp.keys())
    partition = int(len(docs_list) * .8)

    #train_docs = docs_list[0:partition]
    test_docs = docs_list[partition:]

    # Load in solution values
    solution = {}
    docs_list = set(test_docs)
    with open("../data/pm_doc_term_counts.csv", "r") as handle:
        for line in handle:
            line = line.strip("\n").split(",")
            if line[0] in docs_list:
                # Only use samples indexed with MeSH terms
                terms = [term for term in line[1:] if term]
                if terms:
                    solution[line[0]] = terms

    # Build training/test data, ensure good solution data is available
    # Solution data is not always available because documents may not be
    # indexed - even though obviously some of their references have been indexed
#    train_freqs = {}
#    for doc in train_docs:
#        if doc in solution.keys():
#            train_freqs[doc] = temp[doc]

    test_freqs = {}
    for doc in test_docs:
        if doc in solution.keys():
            test_freqs[doc] = temp[doc]  

    import time
    start = time.perf_counter()
    #default_thresh = train(train_freqs, solution, logger)
    print(f"elapsed: {time.perf_counter() - start}")
    
    start = time.perf_counter()
    preds, f1 = predict(test_freqs, default_thresh, solution)
    print(f"elapsed: {time.perf_counter() - start}")
    
if __name__ == "__main__":
    main()
