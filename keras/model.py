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
    mod.add(Dense(29351, activation="sigmoid"))
    mod.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    return mod

@jit
def count_metrics(y, y_hat, threshold):
    true_pos = 0
    false_pos = 0
    false_neg = 0

    n_cols = len(y[0])
    n_rows = len(y)

    for row in range(n_rows):
        for col in range(n_cols):
            y_hat_val = y_hat[row][col]
            y_val = y[row][col]
            
            if y_hat_val > threshold and y_val == 1.0:
                true_pos += 1
            elif y_hat_val <= threshold and y_val == 1.0:
                false_neg += 1
            elif y_hat_val > threshold and y_val == 0.0:
                false_pos += 1
            
    return true_pos, false_pos, false_neg

def test(weights_fp, logger, threshold, mod=get_model(2048)):
    mod.load_weights(weights_fp)

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
        
        tp_temp, fp_temp, fn_temp = count_metrics(y, y_hat, threshold)
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

if __name__ == "__main__":
    logger = get_logger()
    
    ids_list = []
    with open("train_ids", "r") as handle:
        for line in handle:
            ids_list.append(line.strip("\n"))
    
    epochs = 10
    batch_size = 16
    training_generator = DataGen(ids_list[:40000], batch_size) 
    model_code = "current_test"
    fp = f"weights.{model_code}.hdf5"   
    
    #dims = [100, 200, 400, 800, 1600, 2000, 3200]
    # 5000 current best, excluding now for testing
    dims = [8400, 10000, 12000, 14000]
    for dim in dims:
        logger.info(f"training model with {dim} dimension dense")
        mod = get_model(dim)
        mod.fit_generator(generator=training_generator, use_multiprocessing=True, workers=4,
                epochs=epochs)
        mod.save_weights(fp)
        
        test(fp, logger, 0.24, mod)    
        tf.keras.backend.clear_session()
