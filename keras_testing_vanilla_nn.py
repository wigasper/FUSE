import json
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("nn_params_results.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

print("working on input")
subset = []
# load in subset terms list
with open("./data/subset_terms_list", "r") as handle:
    for line in handle:
        subset.append(line.strip("\n"))
subset = set(subset)
        
with open("./data/term_freqs_rev_3_all_terms.json", "r") as handle:
    temp = json.load(handle)

# Get UIDs in a list - for use in building arrays
uids = []
with open("./data/mesh_data.tab", "r") as handle:
    for line in handle:
        line = line.strip("\n").split("\t")
        if line[0] in subset:
            uids.append(line[0])

docs_list = list(temp.keys())
docs_list = docs_list[:160000]
partition = int(len(docs_list) * .9)

train_docs = docs_list[0:partition]
test_docs = docs_list[partition:]

train_freqs = {}
for doc in train_docs:
    train_freqs[doc] = temp[doc]

test_freqs = {}
for doc in test_docs:
    test_freqs[doc] = temp[doc]

# Load in solution values - only for the docs that we need
# Change to set for quick lookup
docs_list = set(docs_list)
solution = {}
with open("./data/pm_doc_term_counts.csv", "r") as handle:
    for line in handle:
        line = line.strip("\n").split(",")
        if line[0] in docs_list:
            terms = [term for term in line[1:] if term in subset]
            if terms:
                solution[line[0]] = terms

train_docs = [doc for doc in train_docs if doc in solution.keys()]
test_docs = [doc for doc in test_docs if doc in solution.keys()]

x = []
for doc in train_docs:
    row = []
    for uid in uids:
        if uid in train_freqs[doc].keys():
            row.append(train_freqs[doc][uid])
        else:
            row.append(0)
    x.append(row)

x = np.array(x)

y = []
for doc in train_docs:
    row = []
    for uid in uids:
        if uid in solution[doc]:
            row.append(1)
        else:
            row.append(0)
    y.append(row)

y = np.array(y)

x_test = []
for doc in test_docs:
    row = []
    for uid in uids:
        if uid in test_freqs[doc].keys():
            row.append(test_freqs[doc][uid])
        else:
            row.append(0)
    x_test.append(row)

x_test = np.array(x_test)

y_test = []
for doc in test_docs:
    row = []
    for uid in uids:
        if uid in solution[doc]:
            row.append(1)
        else:
            row.append(0)
    y_test.append(row)

y_test = np.array(y_test)
    
print("model compilation")

import json
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

model_code = "vanilla.12"

a = Input(shape=(7221,))
b = Dense(256, activation="relu")(a)
b = Dropout(0.5)(b)
b = Dense(256, activation="relu")(b)
b = Dropout(0.5)(b)
b = Dense(256, activation="relu")(b)
b = Dropout(0.5)(b)
b = Dense(256, activation="relu")(b)
b = Dropout(0.5)(b)
#b = Dense(600, activation="relu")(b)
b = Dense(7221, activation="sigmoid")(b)
model = Model(inputs=a, outputs=b)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

with open(f"{model_code}.config.json", "w") as out:
    json.dump(model.get_config(), out)

model.summary(print_fn=logger.info)

# changed batch size 16 to 8 on aug 9 1905
# now to 32 aug 12 1000
batch_size = 16
epochs = 60

fp = f"weights.{model_code}.hdf5"

checkpoint = ModelCheckpoint(fp, monitor="val_loss", verbose=1, save_best_only=True, mode="min")

early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

callbacks_list = [checkpoint, early]

print("starting fit")

model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)

model.load_weights(fp)
##
y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

true_pos = 0
false_pos = 0
false_neg = 0
true_neg = 0

from tqdm import tqdm

for r_idx, row in tqdm(enumerate(y_pred)):
    for c_idx, col in enumerate(row):
        if y_pred[r_idx][c_idx] == 1 and y_test[r_idx][c_idx] == 1:
            true_pos += 1
        if y_pred[r_idx][c_idx] == 0 and y_test[r_idx][c_idx] == 1:
            false_neg += 1
        if y_pred[r_idx][c_idx] == 1 and y_test[r_idx][c_idx] == 0:
            false_pos +=1
        if y_pred[r_idx][c_idx] == 0 and y_test[r_idx][c_idx] == 0:
            true_neg += 1
            
if true_pos > 0:
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    accurracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    f1 = (2 * precision * recall) / (precision + recall)
else:
    f1 = 0

logger.info(f"F1: {f1}, trained on: {len(train_docs)} samples, weights saved as: {fp}") 
logger.info(f"batch_size: {batch_size}, epochs: {epochs}")
# logger.info("used co-occ feature engineering here")

from notify import notify
notify(f"f1: {f1}")
