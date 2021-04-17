import sys
from collections import defaultdict, Counter
from math import log
#%%
train_path = sys.argv[1]
#input_path = 'train_data/italian.txt'

#def train_hmm(input_path):
file = [i.replace('\n','') for i in open(train_path).readlines()]
#file = set([i.split('/')[0] for k in file for i in k.strip().split()]
transition_count, emission_count = [defaultdict(Counter) for i in range(2)]
tag_count = Counter()

for sentence in file:
    text = [tuple(i.rsplit('/',1)) for i in sentence.split()]
    #text = [(word.lower(), tag) for word, tag in text]
    for i in range(len(text)):
        emission_count[text[i][0]].update([text[i][1]])
        tag_count[text[i][1]] += 1
        if i == len(text)-1:
            transition_count['<s>'].update([text[0][1]])
            transition_count[text[i][1]].update(['<e>'])
        else:
            transition_count[text[i][1]].update([text[i+1][1]])

vocab_size = sum(tag_count.values())

[transition_count[tag].update({newtag: 0}) for newtag in tag_count.keys() for tag in transition_count
 if newtag not in transition_count[tag]]

logtransition = {k1: Counter({k2: log(v2+1/(sum(v1.values())+vocab_size)) for k2, v2 in v1.items()}) for k1, v1 in
                 transition_count.items()}

logemission = {k1: Counter({k2: log(v2/tag_count[k2]) for k2, v2 in v1.items()}) for k1, v1 in emission_count.items()}
#%%
with open('hmmmodel.txt', 'w+') as f:
    f.write(str(tag_count) + '\n')
    f.write(str(logtransition) + '\n')
    f.write(str(logemission))