import pandas as pd
from tqdm import tqdm
import pickle

columns = ["node_id"]

for i in range(128):
    columns.append("dim"+str(i+1))

emb = pd.read_csv('embeddings/node2vec/blogcat.emb', delim_whitespace=True, names=columns).sort_values(by=['node_id'], ascending=True)
emb = emb.sample(2710)

dimensions = []
for i in range(128):
    dimensions.append("dim"+str(i+1))

nodes = []
for index, row in emb.iterrows():
    nodes.append(row["node_id"])

pairs = []
for node1 in nodes:
    for node2 in nodes:
        pairs.append([node1, node2, 0])

pairs = pd.DataFrame.from_records(pairs)
pairs.columns=["node1", "node2", "edge"]

node_embeddings = {}
for index, row in emb.iterrows():
    node_embeddings[row["node_id"]] = [row[dim] for dim in dimensions]

edges = pd.read_csv('edgelists/blogcat.edgelist', delim_whitespace=True, names=["node1", "node2"])

for index_edge, row_edge in tqdm(edges.iterrows(), total=edges.shape[0]):
    pair_row = pairs.loc[(pairs['node1'] == row_edge['node1']) & (pairs['node2'] == row_edge['node2'])]
    pairs.loc[pair_row.index, "edge"] = 1

labeled_dataset = pairs

labeled_dataset['node1'] = labeled_dataset['node1'].map(node_embeddings)
labeled_dataset['node2'] = labeled_dataset['node2'].map(node_embeddings)

with open('blogcat-labelled.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(labeled_dataset,f)