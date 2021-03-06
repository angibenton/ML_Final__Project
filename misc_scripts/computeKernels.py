import kernel
import pandas as pd


file = pd.read_csv('./data/p2_train_parsed.csv', index_col=False)

strings = file['tweet'].tolist()
conlls = file['CoNLL'].tolist()

tweets = tuple(zip(strings, conlls))

k = kernel.TFIDFSubgraphsKernel(X=tweets)

p = k.compute_kernel_matrix(X=tweets)

df = pd.DataFrame(data = p.astype(float))

df.to_csv('./data/p2_train_subgraphs_matrix_tfidf.csv', index = False)

print(p)
