import logging

import numpy as np
from tqdm import tqdm

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

def get_model():
    model = keras.Sequential()
    model.add(Dense(2048, activation="relu", input_dim=29351))
    model.add(Dropout(0.1))
    model.add(Dense(29351, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    return model

# TODO: this needs work
def test(fp):
    model = get_model()

    test_ids = []
    with open("test_ids", "r") as handle:
        for line in handle:
            test_ids.append(line.strip("\n"))

    x_test = np.empty((2000,29351))
    y_test = np.empty((2000,29351))

    for idx, test_id in enumerate(test_ids[:2000]):
        x_test[idx,] = np.load(f"data/{test_id}_x.npy")
        y_test[idx,] = np.load(f"data/{test_id}_y.npy")

    model.load_weights(fp)

    y_hat = model.predict(x_test)
    y_hat = np.round(y_hat)

    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    # TODO: this is terribly slow
    for r_idx, row in tqdm(enumerate(y_hat)):
        for c_idx, col in enumerate(row):
            if y_hat[r_idx][c_idx] == 1 and y_test[r_idx][c_idx] == 1:
                true_pos += 1
            if y_hat[r_idx][c_idx] == 0 and y_test[r_idx][c_idx] == 1:
                false_neg += 1
            if y_hat[r_idx][c_idx] == 1 and y_test[r_idx][c_idx] == 0:
                false_pos +=1
            if y_hat[r_idx][c_idx] == 1 and y_test[r_idx][c_idx] == 0:
                true_neg += 1

    if true_pos > 0:
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        accurracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0

    print(f"F1: {f1}, precision: {precision}, recall: {recall}")
    #print(f"trained on: {len(train_docs)} samples, weights saved as: {fp}")
    #print(f"batch_size: {batch_size}, epochs: {epochs}")

if __name__ == "__main__":
    logger = get_logger()
    
    ids_list = []
    with open("train_ids", "r") as handle:
        for line in handle:
            ids_list.append(line.strip("\n"))

    training_generator = DataGen(ids_list)

    model_code = "train_gen_test"

    model = get_model()
    
    batch_size = 16
    epochs = 2

    fp = f"weights.{model_code}.hdf5"
    
    # was val_loss
    checkpoint = ModelCheckpoint(fp, monitor="loss", verbose=1, save_best_only=True, 
                                    mode="min")

    early = EarlyStopping(monitor="loss", mode="min", patience=20)

    callbacks_list = [checkpoint, early]

    model.fit_generator(generator=training_generator, use_multiprocessing=True, workers=4, 
            epochs=epochs, callbacks=callbacks_list)
    #model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)

        
