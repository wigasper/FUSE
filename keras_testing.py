import json
import numpy as np

print("working on input")
subset = []
# load in subset terms list
with open("./data/subset_terms_list", "r") as handle:
    for line in handle:
        subset.append(line.strip("\n"))
subset = set(subset)
        
# Load in term frequencies
#with open("./data/term_freqs_rev_2_all_terms.json", "r") as handle:
#    temp = json.load(handle)
 
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
docs_list = docs_list[:40000]
partition = int(len(docs_list) * .8)

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
            solution[line[0]] = [term for term in line[1:] if term in subset]

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

x = x.reshape(30430, 7221, 1)
#x = x.reshape(1, 7221, 45659)

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

x_test = x_test.reshape(7604, 7221, 1)
#x_test = x_test.reshape(1, 7221, 45659)

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

from keras.models import Model
from keras.layers import Dense, Input, LSTM, Bidirectional, GlobalMaxPool1D, Dropout, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint

a = Input(shape=(7221, 1,))
#b = Embedding(7221, 256)(a)
b = Bidirectional(LSTM(256, return_sequences=True))(a)
b = GlobalMaxPool1D()(b)
b = Dropout(0.1)(b)
b = Dense(256, activation="relu")(b)
b = Dropout(0.1)(b)
b = Dense(7221, activation="sigmoid")(b)
model = Model(inputs=a, outputs=b)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

batch_size = 32
epochs = 2

fp = "weights.hdf5"

checkpoint = ModelCheckpoint(fp, monitor="val_loss", verbose=1, save_best_only=True, mode="min")

early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

callbacks_list = [checkpoint, early]

print("starting fit")

model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)

#model.load_weights(fp)
##
#y_pred = model.predict(x_test)
#y_pred_rnd = np.round(y_pred)

"""
for row in y_pred:
    for col in row:
"""
#t1 = y_pred[0]
#col = 0
#max_c = 0
#for index, column in enumerate(t1):
#    if t1[index] > max_c:
#        max_c = t1[index]
#        col = index