import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

labeled_dataset = pd.read_pickle('cora-labelled.pkl', compression='infer')
labeled_sample = labeled_dataset.sample(1000000)

training_set, validation_set = train_test_split(labeled_sample, test_size = 0.5, random_state = 21)

X_train = training_set[["node1", "node2"]]
Y_train = training_set[["edge"]]
X_val = validation_set[["node1", "node2"]]
Y_val = validation_set[["edge"]]

print("Splitting...")
X_train_split = pd.concat([pd.DataFrame(X_train['node1'].to_list()),pd.DataFrame(X_train['node2'].to_list())], axis=1)
with open('X_train_split.pkl', 'wb') as f:
    pickle.dump(X_train_split,f)

logisticRegr = LogisticRegression()
logisticRegr.fit(X_train_split, Y_train.values.ravel())

X_val_split = pd.concat([pd.DataFrame(X_val['node1'].to_list()),pd.DataFrame(X_val['node2'].to_list())], axis=1)
with open('X_val_split.pkl', 'wb') as f:
    pickle.dump(X_val_split,f)

Y_pred = logisticRegr.predict(X_val_split)

with open("cora_n2v_scores.txt", "w") as f:
    # Writing data to a file
    f.write("Macro F1: ")
    f.write(str(f1_score(Y_val, Y_pred, average='macro')))
    f.write("\n")
    f.write("Micro F1: ")
    f.write(str(f1_score(Y_val, Y_pred, average='micro')))  
