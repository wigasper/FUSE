import json
import numpy as np
import time

print("working on input")
subset = []
# load in subset terms list
with open("./data/subset_terms_list", "r") as handle:
    for line in handle:
        subset.append(line.strip("\n"))
subset = set(subset)
        
# Load in term frequencies
with open("./data/term_freqs_rev_2_all_terms.json", "r") as handle:
    temp = json.load(handle)
 
# Get UIDs in a list - for use in building arrays
uids = []
with open("./data/mesh_data.tab", "r") as handle:
    for line in handle:
        line = line.strip("\n").split("\t")
        if line[0] in subset:
            uids.append(line[0])

docs_list = list(temp.keys())

train_docs = docs_list[0:3000]

train_freqs = {}
for doc in train_docs:
    train_freqs[doc] = temp[doc]

# Load in solution values - only for the docs that we need
# Change to set for quick lookup
docs_list = set(docs_list)
solution = {}
with open("./data/pm_doc_term_counts.csv", "r") as handle:
    for line in handle:
        line = line.strip("\n").split(",")
        if line[0] in docs_list:            
            solution[line[0]] = [term for term in line[1:] if term in subset]

train_docs = [doc for doc in train_docs if doc in solution.keys()]

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

print("model compilation")

from keras.models import Model
from keras.layers import Dense, Input, LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

a = Input(shape=(7221,))
b = Bidirectional(LSTM(50, return_sequences=True))(a)
b = GlobalMaxPool1D()(b)
b = Dropout(0.1)(b)
b = Dense(50, activation="relu")(b)
b = Dropout(0.1)(b)
b = Dense(7221, activation="sigmoid")(b)
model = Model(inputs=a, outputs=b)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

batch_size = 32
epochs = 5

fp = "weights.hdf5"

checkpoint = ModelCheckpoint(fp, monitor="val_loss", verbose=1, save_best_only=True, mode="min")

early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

callbacks_list = [checkpoint, early]

print("starting fit")

model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)
