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
def count_metrics(y, y_hat, thresholds):
    true_pos = 0
    false_pos = 0
    false_neg = 0

    n_cols = len(y[0])
    n_rows = len(y)

    for row in range(n_rows):
        for col in range(n_cols):
            y_hat_val = y_hat[row][col]
            y_val = y[row][col]
            threshold = thresholds[col]

            if y_hat_val > threshold and y_val == 1.0:
                true_pos += 1
            elif y_hat_val <= threshold and y_val == 1.0:
                false_neg += 1
            elif y_hat_val > threshold and y_val == 0.0:
                false_pos += 1
            
    return true_pos, false_pos, false_neg

def test(weights_fp, logger, threshold, mod=get_model(2048), decision_thresholds):
    mod.load_weights(weights_fp)

    # this is a dict, need to make a list for better numba usage
    thresholds = [decision_thresholds[resp] for resp in range(len(decision_thresholds))]

    test_ids = []
    with open("test_ids", "r") as handle:
        for line in handle:
            test_ids.append(line.strip("\n"))

    true_pos = 0
    false_pos = 0
    false_neg = 0
    
    batch_size = 16
    test_gen = DataGen(test_ids, batch_size)

    for batch in tqdm(test_gen):
        x = batch[0]
        y = batch[1]

        y_hat = mod.predict_on_batch(x)
        
        tp_temp, fp_temp, fn_temp = count_metrics(y, y_hat, thresholds)
        true_pos += tp_temp
        false_pos += fp_temp
        false_neg += fn_temp
           
    if true_pos > 0:
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0

    logger.info(f"F1: {f1}, precision: {precision}, recall: {recall}, threshold: {threshold}")

@jit
def get_f1(threshold, y, yhat):
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

def find_decision_thresholds(validation_ids, weights_fp, logger, mod):
    # first need to save arrays with y and yhat
    mod.load_weights(weights_fp)
    batch_size = 16
    test_gen = DataGen(validation_ids, batch_size)
    default_thresh = 0.33
    
    batch_num = 0
    batch_ids = []

    logger.info("Generating predictions and saving")
    for batch in tqdm(test_gen):
        x = batch[0]
        y = batch[1]

        y_hat = mod.predict_on_batch(x)
        
        np.save(f"y/{batch_num}_y.npy", y)
        np.save(f"y/{batch_num}_y_hat.npy", y_hat)
        batch_ids.append(batch_num)
        batch_num += 1
    
    responses = {num: default_thresh for num in range(29351)}

    thresholds = [x * .01 for x in range(100)]

    logger.info("Determining optimal thresholds for each response")
    for resp in tqdm(responses):
        ys = []
        y_hats = []

        for batch in batch_ids:
            y = np.load(f"y/{batch_num}_y.npy")
            y_hat = np.load(f"y/{batch_num}_y_hat.npy") 
            
            ys.extend([val for val in y[:,resp]])
            y_hats.extend([val for val in y_hat[:,resp]])
        
        f1s = []
        for idx, threshold in enumerate(thresholds):
            f1s.append(get_f1(threshold, y, y_hat))
            # early stopping
            if f1s[-10] > f1s[-1]:
                break

        responses[resp] = thresholds[f1s.index(max(f1s))]

    with open("decision_thresholds.json", "w") as out:
        json.dump(responses, out)

    return responses

if __name__ == "__main__":
    logger = get_logger()
    
    ids_list = []
    with open("train_ids_expanded", "r") as handle:
        for line in handle:
            ids_list.append(line.strip("\n"))
    
    validation_ids = ids_list[400000:]
    ids_list = ids_list[:400000]

    epochs = 30
    batch_size = 16
    training_generator = DataGen(ids_list, batch_size) 
    model_code = "current_test"
    fp = f"weights.{model_code}.hdf5"   
    logger.info(f"training on all ids, expanded training set. {epochs} epochs")

    #dims = [100, 200, 400, 800, 1600, 2000]
    dim = 1600
    #for dim in dims:
    logger.info(f"training model with {dim}, {dim} dimension dense")
    mod = get_model(dim)
    mod.fit(training_generator, use_multiprocessing=True, workers=4,
            epochs=epochs)
    mod.save_weights(fp)
    
    # good default decision threshold is 0.33 
    # move to validation, need to learn an optimal decision threshold for each class
    find_decision_thresholds(validation_ids, weights_fp, logger, mod)

    test(weights_fp, logger, threshold, mod, decision_thresholds)
