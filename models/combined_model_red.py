import json
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("combined_results.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

print("working on input")
subset = []
# load in subset terms list
with open("../data/subset_terms_list", "r") as handle:
    for line in handle:
        subset.append(line.strip("\n"))
subset = set(subset)
        
with open("../data/term_freqs_rev_3_all_terms.json", "r") as handle:
    temp = json.load(handle)

top_500 = []
with open("../data/top_500_terms", "r") as handle:
    for line in handle:
        top_500.append(line.strip("\n"))
top_500_set = set(top_500)

# Get UIDs in a list - for use in building arrays
uids = []
infreq_subset = []
with open("../data/mesh_data.tab", "r") as handle:
    for line in handle:
        line = line.strip("\n").split("\t")
        if line[0] in subset:
            uids.append(line[0])
        elif line[0] not in top_500_set:
            infreq_subset.append(line[0])

infreq_subset = set(infreq_subset)

docs_list = list(temp.keys())
partition = int(len(docs_list) * .9)

test_docs = docs_list[partition:]

test_freqs = {}
for doc in test_docs:
    test_freqs[doc] = temp[doc]

# Load in solution values - only for the docs that we need
# Change to set for quick lookup
docs_list = set(docs_list[partition:])
solution = {}
with open("../data/pm_doc_term_counts.csv", "r") as handle:
    for line in handle:
        line = line.strip("\n").split(",")
        if line[0] in docs_list:
            terms = [term for term in line[1:]]
            if terms:
                solution[line[0]] = terms

test_docs = [doc for doc in test_docs if doc in solution.keys()]

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
    for uid in top_500:
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

model_code = "reduced.resp.1"

a = Input(shape=(7221,))
b = Dense(512, activation="relu")(a)
b = Dropout(0.1)(b)
#b = Dense(256, activation="relu")(b)
#b = Dropout(0.5)(b)
#b = Dense(256, activation="relu")(b)
#b = Dropout(0.5)(b)
#b = Dense(256, activation="relu")(b)
#b = Dropout(0.5)(b)
#b = Dense(600, activation="relu")(b)
b = Dense(500, activation="sigmoid")(b)
model = Model(inputs=a, outputs=b)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

#with open(f"{model_code}.config.json", "w") as out:
#    json.dump(model.get_config(), out)

model.summary(print_fn=logger.info)

# changed batch size 16 to 8 on aug 9 1905
# now to 32 aug 12 1000
batch_size = 16
epochs = 60

fp = f"../weights.{model_code}.hdf5"

checkpoint = ModelCheckpoint(fp, monitor="val_loss", verbose=1, save_best_only=True, mode="min")

early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

callbacks_list = [checkpoint, early]

print("starting fit")

model.load_weights(fp)

# Predict infrequent terms with desc. thresh
thresh = 0.011

predictions = {}
for doc in test_freqs:
    predictions[doc] = []
    for key, val in test_freqs[doc].items():
        if val > thresh and key in infreq_subset:
            predictions[doc].append(key)

# Predict frequent terms with NN
y_pred = model.predict(x_test)
y_pred = np.round(y_pred)

# add y_preds to dict
for r_idx, row in enumerate(y_pred):
    for c_idx, col in enumerate(row):
        if y_pred[r_idx][c_idx] == 1:
            predictions[test_docs[r_idx]].append(top_500[c_idx])


true_pos = 0
false_pos = 0
false_neg = 0
true_neg = 0

for pmid in predictions:
    true_pos += len([pred for pred in predictions[pmid] if pred in solution[pmid]])
    false_pos += len([pred for pred in predictions[pmid] if pred not in solution[pmid]])
    false_neg += len([sol for sol in solution[pmid] if sol not in predictions[pmid]])

if true_pos > 0:
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    accurracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    f1 = (2 * precision * recall) / (precision + recall)
else:
    f1 = 0

logger.info(f"F1: {f1}, used {model_code}") 
# logger.info("used co-occ feature engineering here")

from notify import notify
notify(f"f1: {f1}")
