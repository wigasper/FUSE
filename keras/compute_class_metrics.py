import json
import logging

from numba import jit
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from datagen import DataGen

def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("nn_params_results.log")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

def get_model(dim):
    mod = keras.Sequential()
    mod.add(Dense(dim, activation="relu", input_dim=29351))
    mod.add(Dropout(0.1))
    mod.add(Dense(dim, activation="relu"))
    mod.add(Dropout(0.1))
    mod.add(Dense(29351, activation="sigmoid"))
    mod.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    return mod

@jit
def count_metrics(y, y_hat, class_index, threshold):
    true_pos = 0
    false_pos = 0
    false_neg = 0

    #n_cols = len(y[0])
    n_rows = len(y)
    col = class_index
    #threshold = 0.33

    for row in range(n_rows):
        #for col in range(n_cols):
        y_hat_val = y_hat[row][col]
        y_val = y[row][col]
        #threshold = thresholds[col]

        if y_hat_val > threshold and y_val == 1.0:
            true_pos += 1
        elif y_hat_val <= threshold and y_val == 1.0:
            false_neg += 1
        elif y_hat_val > threshold and y_val == 0.0:
            false_pos += 1
        
    return true_pos, false_pos, false_neg

def test(weights_fp, logger, mod, threshold):
    mod.load_weights(weights_fp)

    test_ids = []
    with open("test_ids", "r") as handle:
        for line in handle:
            test_ids.append(line.strip("\n"))
    
    class_metrics = {num: {"true_pos": 0, "false_pos": 0, "false_neg": 0} for num in range(29351)}
    
    batch_size = 16
    test_gen = DataGen(test_ids, batch_size)

    for batch in tqdm(test_gen):
        x = batch[0]
        y = batch[1]

        y_hat = mod.predict_on_batch(x)
        
        for class_index in class_metrics:
            metrics = count_metrics(y, y_hat, class_index, threshold)
            class_metrics[class_index]["true_pos"] += metrics[0]
            class_metrics[class_index]["false_pos"] += metrics[1]
            class_metrics[class_index]["false_neg"] += metrics[2]
        
   
    f1s_by_class = {num: 0.0 for num in range(29351)}
    for class_index in class_metrics:
        if class_metrics[class_index]["true_pos"] > 0:
            true_pos = class_metrics[class_index]["true_pos"]
            false_pos = class_metrics[class_index]["false_pos"]
            false_neg = class_metrics[class_index]["false_neg"]
            
            precision = true_pos / (true_pos + false_pos)
            recall = true_pos / (true_pos + false_neg)
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        f1s_by_class[class_index] = f1
    
    return f1s_by_class

@jit
def get_f1(threshold, y, y_hat):
    true_pos = 0
    false_pos = 0
    false_neg = 0
    
    y_len = len(y)

    for idx in range(y_len):
        y_hat_val = y_hat[idx]
        y_val = y[idx]
            
        if y_hat_val > threshold and y_val == 1.0:
            true_pos += 1
        elif y_hat_val <= threshold and y_val == 1.0:
            false_neg += 1
        elif y_hat_val > threshold and y_val == 0.0:
            false_pos += 1

    if true_pos == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f1 = (2 * precision * recall) / (precision + recall)

    return f1


if __name__ == "__main__":
    logger = get_logger()
    
    epochs = 30
    batch_size = 16
    model_code = f"current_best"
    fp = f"weights.{model_code}.hdf5"   
    dim = 1600
    threshold = 0.33
    mod = get_model(dim)

    f1s_by_class = test(fp, logger, mod, threshold)

    # need to map to mesh UIDs
    uids = []
    with open("../data/mesh_data.tab", "r") as handle:
        for line in handle:
            line = line.strip("\n").split("\t")
            uids.append(line[0])
    
    f1s_out = {}
    for index, uid in enumerate(uids):
        f1s_out[uid] = f1s_by_class[index]

    with open("class_f1s.json", "w") as out:
        json.dump(f1s_out, out)
