from config import *
import pickle
import itertools

def read(file_path):
    with open(file_path) as f:
        cases = []
        document = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx, line = line.split(' ', 1)
            # print(idx, 'sentence:', line)
            if 'XXXXX' in line:
                q, a, _, candidates = line.split('\t')
                q = q.split()
                candidates = candidates.split('|')
                # print(line,'_', q,a,candidates)
                cases.append((document, q, a, candidates))
                document = []
            else:
                words = line.split()
                document.extend(words)
    print('Number of samples in {} is {}'.format(file_path, len(cases)))
    return cases


def BuildVoc(train, valid):
    # Big Notice:itertools.chain treat ['the',['the']] as different data, 1st 'the' will be divided into 't','h','e',
    # ['the'] will be not, as it treat words within lists as inidividual word, treat string as character division
    # So, it would be added [[]] to a cuz it'a string-->Wrong, it has not to
    # Because d+q+[a]+c is [sample1,....]->sample:[w1,..,wn(d),...,w(a),...]
    alldata = train + valid
    voc = set(itertools.chain(*(d+q+[a]+c for d,q,a,c in alldata)))
    vocab_size = len(voc) + 1
    word2idx = {w: idx+1 for idx, w in enumerate(voc)}
    word2idx.update({'pad': 0})
    idx2word = {v: k for k,v in word2idx.items()}

    # documentLen, queryLen, candidateLen
    documentLen = max([len(d) for d,_,_,_ in alldata])
    queryLen = max([len(q) for _,q,_,_ in alldata])
    candidateLen = max([len(c) for _,_,_,c in alldata])
    return vocab_size, voc, word2idx, idx2word, documentLen, queryLen, candidateLen

def PAD(seq, length):
    if len(seq) < length:
        seq.extend([0]*(length - len(seq)))
    return seq

def TransForm(word2idx, data, dl, ql):
    idx_data = []
    for d, q, a, c in data:
        d = [word2idx[w] for w in d]
        q = [word2idx[w] for w in q]
        a = word2idx[a]
        c = [word2idx[w] for w in c]
        d = PAD(d, dl)
        q = PAD(q, ql)
        idx_data.append((d, q, a, c))
    return idx_data

def Write():
    # Store data
    pickle.dump(train_data, open(train_idx_path, 'wb'))
    pickle.dump(valid_data, open(valid_idx_path, 'wb'))
    pickle.dump((vocab_size, vocab, word2idx, idx2word, dl, ql, cl), open(vocab_path, 'wb'))

    # Wirte all configures into config.py
    with open('./config.py', 'a') as cf:
        cf.write('VOCAB_SIZE = ' + str(vocab_size) + '\n')
        cf.write('TRAIN_SIZE = ' + str(len(train_data)) + '\n')
        cf.write('VALID_SIZE = ' + str(len(valid_data)) + '\n')
    return

if __name__ == '__main__':
    # (1) [document, query, answer, candidates]
    print("Start pre-processing text data!")
    train = read(cbt_ne_train)   # Number of samples in train set: 108719
    valid = read(cbt_ne_valid)    # Number of samples in test set: 2000

    # (2) build training data dictionary
    vocab_size, vocab, word2idx, idx2word, dl, ql, cl = BuildVoc(train, valid)

    # (3) idx ver. [document, query, answer, candidates]
    train_data = TransForm(word2idx, train, dl, ql)
    valid_data = TransForm(word2idx, valid, dl, ql)

    # Write data to store
    Write()
    print("Vocabulary size: {}, max document size: {}, "
          "max question size: {}, max candidate size: {}".format(vocab_size, dl, ql, cl))
    # Vocabulary size: 61200, max document size: 1338,
    # max question size: 210, max candidate size: 11

    # valid candidates len: Counter({10: 2000}), trian: Counter({10: 108707, 11: 12})
    print("Done!")
