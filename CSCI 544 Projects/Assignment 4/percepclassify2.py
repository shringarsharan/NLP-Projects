import sys, re, os.path, string
from collections import Counter, defaultdict
from itertools import count
from math import log
#%%
model_path = sys.argv[1]
test_path = sys.argv[2]
files = [os.path.join(root, file) for root, dir, files in os.walk(test_path) for file in files if not dir]

classes = ['positive', 'negative', 'truthful', 'deceptive']
stopwords = [i.replace('\n', '') for i in list(open('stopwords3.txt', 'r'))]
stopwords = [i for i in stopwords if i not in ['not']]

with open(model_path, 'r') as f:
    weights = eval(f.readline().replace('\n', ''))
    bias = eval(f.readline().replace('\n', ''))
    train_vocab = eval(f.readline().replace('\n', ''))
# %%
def classification(files, stopwords, weights, bias, train_vocab):
    classify = defaultdict(list)
    for file in files:
        # Classification
        text = open(file, 'r').readline().replace('\n','').strip().lower()
        text = text.translate(str.maketrans('', '', string.punctuation.replace('\'','')+string.digits)).strip()
        tokens = [i for i in text.split() if i not in stopwords]
        activation = [0,0]
        for clf in range(2):
            activation[clf] = sum(weights[word][clf]*train_vocab[word] for word in tokens if word in train_vocab) + bias[clf]
            if clf == 0:
                classify[file].append('positive' if activation[clf] > 0 else 'negative')
            elif clf == 1:
                classify[file].append('truthful' if activation[clf] > 0 else 'deceptive')
    return classify
#%%
output = classification(files, stopwords, weights, bias, train_vocab)

with open('percepoutput.txt','w') as f:
    for file in output:
        f.write(f'{output[file][1]} {output[file][0]} {file}' + '\n')