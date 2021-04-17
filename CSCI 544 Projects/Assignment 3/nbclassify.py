import sys, re, os.path, string
from collections import Counter, defaultdict
from itertools import count
from math import log
#%%
def lnprob_x_y(token, category, all_vocab):
    category = eval(category)
    return log((category[token] + 1)/(sum(category.values()) + len(all_vocab)))

def lnprob_y_x(tokens, category, lnp_y, all_vocab):
    prob_y_x = lnp_y[category] + sum(lnprob_x_y(token, category, all_vocab) for token in tokens)
    return prob_y_x
#%%
test_path = sys.argv[1]
files = [os.path.join(root, file) for root,dir,files in os.walk(test_path) for file in files if not dir]


classes = ['positive', 'negative', 'truthful', 'deceptive']
stopwords = [i.replace('\n','') for i in list(open('stopwords2.txt','r'))]

with open('nbmodel.txt','r') as f:
    lnp_y = eval(f.readline().replace('\n',''))
    positive = eval(f.readline().replace('\n',''))
    negative = eval(f.readline().replace('\n',''))
    truthful = eval(f.readline().replace('\n',''))
    deceptive = eval(f.readline().replace('\n',''))
    train_vocab = eval(f.readline().replace('\n',''))
    
print(lnp_y)    

print({k: sum(eval(k).values()) for k in classes})
#%%
def classification(files, stopwords):
    classify = defaultdict(list)
    for file in files:
        # Classification
        text = open(file, 'r').readline().replace('\n','').strip().lower()
        text = text.translate(str.maketrans('', '', string.punctuation+string.digits)).strip()
        tokens = [i for i in text.split() if i not in stopwords]
        classify[file].append('truthful' if lnprob_y_x(tokens, 'truthful', lnp_y, train_vocab) >
                                            lnprob_y_x(tokens, 'deceptive', lnp_y, train_vocab) else 'deceptive')
        classify[file].append('positive' if lnprob_y_x(tokens, 'positive', lnp_y, train_vocab) >
                                            lnprob_y_x(tokens, 'negative', lnp_y, train_vocab) else 'negative')
    return classify
#%%
output = classification(files, stopwords)


with open('nboutput.txt','w') as f:
    for file in output:
        f.write(f'{output[file][0]} {output[file][1]} {file}' + '\n')