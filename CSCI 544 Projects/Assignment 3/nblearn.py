import sys, re, os.path, string
from collections import Counter, defaultdict
from itertools import count
from math import log

def opposite_class(cls):
    pos_neg = ['positive', 'negative']
    tru_dec = ['truthful', 'deceptive']
    if cls in pos_neg:
        return [i for i in pos_neg if i not in cls][0]
    else:
        return [i for i in tru_dec if i not in cls][0]
#%%    

train_path = sys.argv[1]
classes = ['positive', 'negative', 'truthful', 'deceptive']
files = [os.path.join(root, file) for root,dir,files in os.walk(train_path) for file in files if not dir]

stopwords = [i.replace('\n','') for i in list(open('stopwords2.txt','r'))]
train_vocab, positive, negative, truthful, deceptive = [Counter() for i in range(5)]
prior = Counter()

for file in files:
    text = open(file, 'r').readline().replace('\n','').strip().lower()
    text = text.translate(str.maketrans('', '', string.punctuation+string.digits)).strip()
    tokens = [i for i in text.split() if i not in stopwords]
    train_vocab.update(tokens)
    for cls in classes:
        if cls in file:
            eval(cls).update(tokens)
            prior[cls] += 1

#%%
print({k: sum(eval(k).values()) for k in classes})
# positive = Counter(dict(filter(lambda x: x[1] > 5, positive.items())))
# negative = Counter(dict(filter(lambda x: x[1] > 5, negative.items())))
# truthful = Counter(dict(filter(lambda x: x[1] > 5, truthful.items())))
# deceptive = Counter(dict(filter(lambda x: x[1] > 5, deceptive.items())))
# print({k: sum(eval(k).values()) for k in classes})

#all_vocab = set(list(positive.keys()) + list(negative.keys()) + list(truthful.keys()) + list(deceptive.keys()))
#%%
#lnp_y = {'positive':log(len(positive)), 'negative':log(len(negative)), 'truthful':log(len(truthful)),
#         'deceptive':log(len(deceptive))}
lnp_y = {cls: log(prior[cls]/(prior[cls]+prior[opposite_class(cls)])) for cls in classes}
print(prior)

#%%
with open('nbmodel.txt', 'w') as f:
    f.write(str(lnp_y)+'\n')
    f.write(str(positive)+'\n')
    f.write(str(negative)+'\n')
    f.write(str(truthful) + '\n')
    f.write(str(deceptive) + '\n')
    f.write(str(train_vocab))
f.close()