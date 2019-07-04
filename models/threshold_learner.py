#!/usr/bin/env python3
import json
import time
import logging
import argparse
from multiprocessing import Process, Queue
from copy import deepcopy

# MP worker that calculates the optimately discrimination threshold
# for a single UID
def uid_worker(work_queue, write_queue, term_freqs, solution):
    # optimize this
    thresholds = [x * .005 for x in range(0,200)]

    while True:
        uid = work_queue.get()
        if uid is None:
            break
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

        max_thresh = thresholds[f1s.index(max(f1s))]
        write_queue.put((uid, max_thresh))

# MP worker that writes results to the dict
def dict_writer(write_queue, completed_queue, uids):
    thresholds_dict = {uid: 0 for uid in uids}
    while True:
        result = write_queue.get()
        if result is None:
            completed_queue.put(thresholds_dict)
            break
        thresholds_dict[result[0]] = result[1]

def learn_default_threshold(term_freqs, solution):
    thresholds = [x * .005 for x in range(0,200)]
    
    predictions = {}
    precisions = []
    recalls = []
    f1s = []
    
    # Run the model for all thresholds
    for thresh in thresholds:
        # Predict
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
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    return thresholds[f1s.index(max(f1s))]

def predict(test_freqs, solution):
    uid_thresholds = {}
    # Load training results
    with open("../data/individual_term_thresholds.csv", "r") as handle:
        for line in handle:
            line = line.strip("\n").split(",")
            uid_thresholds[line[0]] = float(line[1])

    # Test it out
    predictions = {}
            
    # Predict
    for doc in test_freqs.keys():
#        predictions[doc] = [key for key, val in test_freqs[doc].items() if val > uid_thresholds[key]]
	predictions[doc] = [key for key, val in test_freqs[doc].items() if val > .02]
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

    from notify import notify     
    notify(f"all done, precision: {precision}, recall: {recall}, f1: {f1}")

    return predictions

def train(train_freqs, solution, logger):
    # Load in UIDs
    uids = []
    with open("../data/mesh_data.tab", "r") as handle:
        for line in handle:
            line = line.strip("\n").split("\t")
            uids.append(line[0])

    # MP architecture
    num_workers = 6
    # this is sort of ready to increase num writers but currently 
    # hacked together and num_writers should not be more than 1
    # unless dict (from completed queue) joining logic is added at the end
    num_writers = 1
    work_queue = Queue(maxsize=100)
    write_queue = Queue(maxsize=100)
    completed_queue = Queue(maxsize=10)

    writers = [Process(target=dict_writer, args=(write_queue, completed_queue, 
                uids)) for _ in range(num_writers)]

    for writer in writers:
        writer.daemon = True
        writer.start()

    workers = [Process(target=uid_worker, args=(work_queue, write_queue, 
                deepcopy(train_freqs), deepcopy(solution))) for _ in range(num_workers)]

    for worker in workers:
        worker.start()

    counter = 0
    logging_interval = 100
    for uid in uids:
        if counter % logging_interval == 0:
            logger.info(f"{counter} UIDs added to queue")
        work_queue.put(uid)
        counter += 1

    while True:
        if work_queue.empty():
            for _ in range(num_workers):
                work_queue.put(None)
            break
    
    for worker in workers:
        worker.join()

    for writer in writers:
        write_queue.put(None)

    if write_queue.empty():
        uid_thresholds = completed_queue.get()
        for writer in writers:
            writer.join()

    default_thresh = learn_default_threshold(train_freqs, solution)

    for uid in uid_thresholds:
        if uid_thresholds[uid] == 0: # need to learn ceiling here too!!!
            uid_thresholds[uid] = default_thresh

    # Save training results
    with open("../data/individual_term_thresholds.csv", "w") as out:
        for uid in uid_thresholds:
            out.write("".join([uid, ",", str(uid_thresholds[uid]), "\n"]))

def main():
    # Get command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", help="train model", action="store_true")
    parser.add_argument("-p", "--predict", help="predict", action="store_true")
    parser.add_argument("-i", "--input", help="input file path with term freqs for each doc", type=str)
    args = parser.parse_args()

    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("../logs/learn_individ_thresholds_mp.log")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Load in term frequencies and partition
    if args.input:
        with open(args.input, "r") as handle:
            temp = json.load(handle)
    else:
        with open("../data/term_freqs_rev_0.json", "r") as handle:
            temp = json.load(handle)

    docs_list = list(temp.keys())
    partition = int(len(docs_list) * .8)

    train_docs = docs_list[0:partition]
    test_docs = docs_list[partition:]

    # Load in solution values
    solution = {}
    docs_list = set(docs_list)
    with open("../data/pm_doc_term_counts.csv", "r") as handle:
        for line in handle:
            line = line.strip("\n").split(",")
            if line[0] in docs_list:
                solution[line[0]] = line[1:]

    train_freqs = {}
    for doc in train_docs:
        if doc in solution.keys():
            train_freqs[doc] = temp[doc]

    test_freqs = {}
    for doc in test_docs:
        if doc in solution.keys():
            test_freqs[doc] = temp[doc]

    if args.train:
        train(train_freqs, solution, logger)
    
    if args.predict:
        preds = predict(test_freqs, solution)

if __name__ == "__main__":
    main()
