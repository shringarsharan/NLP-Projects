import sys
from collections import defaultdict, Counter
from itertools import product
import numpy as np
#%%
test_path = sys.argv[1]
#test_path = 'test_data/italian.txt'

#def train_hmm(input_path):
file = [i.replace('\n','') for i in open(test_path).readlines()]

with open('hmmmodel.txt', 'r+') as f:
    tag_count = Counter(eval(f.readline().replace('\n', '')))
    logtransition = defaultdict(Counter, eval(f.readline().replace('\n', '')))
    logemission = defaultdict(Counter, eval(f.readline().replace('\n', '')))

all_states = set(logtransition.keys()) - set(['<s>','<e>'])
open_class = dict(tag_count.most_common(8)).keys()
#%%
def hmmdecode2(sentence):
    text = sentence.split()
    path = defaultdict(Counter)
    for i in range(len(text)):
        if logemission[text[i]]:
            EM = logemission.get(text[i],open_class)
        else:
            EM = open_class
        if i == 0:
            [path[i].update({','.join(['<s>', curr_st]): logtransition['<s>'][curr_st] + logemission[text[i]][curr_st]})
             for curr_st in EM if logtransition['<s>'][curr_st] != 0]
        elif i == len(text)-1:
            for curr_st in EM:
                calcMax = Counter({','.join([prev_sts, curr_st]): path[i-1].get(prev_sts,0)+
                              logtransition[prev_sts.rsplit(',', 1)[-1]][curr_st]+
                              logtransition[curr_st]['<e>']+
                              logemission[text[i]][curr_st] for prev_sts in path[i-1]}).most_common(1)
                path[i].update(dict(calcMax))
        else:
            for curr_st in EM:
                calcMax = Counter({','.join([prev_sts, curr_st]): path[i-1].get(prev_sts,0)+
                              logtransition[prev_sts.rsplit(',', 1)[-1]][curr_st]+
                              logemission[text[i]][curr_st] for prev_sts in path[i-1]}).most_common(1)
                path[i].update(dict(calcMax))
        #print(f'{text[i]}: {path[i]}, {EM}\n-----------')
    #print(f'{sentence}---> {path[i]}')
    most_probable_path = path[i].most_common(1)[0][0].split(',',1)[1].split(',')
    return ' '.join(['/'.join(w) for w in zip(text, most_probable_path)])
#%%
#print(hmmdecode2(file[3]))
#%%
with open('hmmoutput.txt','w+') as f:
    for sentence in file:
        output = hmmdecode2(sentence)
        f.write(output + '\n')