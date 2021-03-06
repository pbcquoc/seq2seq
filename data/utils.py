import numpy as np
import json
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu

EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz '
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''
UNK = 'unk'
BOS = 'BOS'
EOS = 'EOS'
PAD = '_'


VOCAB_SIZE = 6000

limit = {
        'maxq' : 20,
        'minq' : 5,
        'maxa' : 20,
        'mina' : 5
        }

fi = 'twitter_en.txt'

def filter(str, whitelist):
  return "".join(c for c in str.lower() if c in whitelist)

def preprocess():
  all_words, ques, ans = [], [], []
  chats = open(fi, 'r').readlines()
  
  filtered = map(lambda str: filter(str, EN_WHITELIST), chats)

  for i in xrange(0, len(chats), 2):
    q, a = filtered[i].split(' '), filtered[i+1].split(' ')
    
    if len(q) > limit['minq'] and len(q) < limit['maxq']  and \
      len(a) > limit['mina'] and len(a) < limit['maxa']:
      missing_q = [PAD]*(limit['maxq'] - len(q))
      missing_a = [PAD]*(limit['maxa'] - len(a)) 

      ques.append(q + missing_q)
      ans.append(['BOS'] + a + ['EOS'] + missing_a)
      all_words += ['BOS', 'EOS'] + a + q + missing_q + missing_a

  # make vocabulary  
  counter = Counter(all_words)
  vocabs = counter.most_common(VOCAB_SIZE)
  common_count = [c[1] for c in vocabs]
  print 1.0*sum(common_count)/len(all_words)
  vocabs = [c[0] for c in vocabs] + [UNK]
  w2idx = dict(zip(vocabs, range(len(vocabs))))
  idx2w = dict(zip(range(len(vocabs)), vocabs))
  
  idx_q = np.zeros([len(ques), limit['maxq']], dtype=np.int32)
  idx_a = np.zeros([len(ans), limit['maxa'] +2], dtype=np.int32)
  # process rare word
  for i in xrange(len(ques)):
    for j in xrange(len(ques[i])):
      if ques[i][j] not in w2idx:
        ques[i][j] = UNK
      idx_q[i, j] = w2idx[ques[i][j]]

    for j in xrange(len(ans[i])):
      if ans[i][j] not in w2idx:
        ans[i][j] = UNK
      idx_a[i, j] = w2idx[ans[i][j]]
  print ques[0], ans[0]
  print idx_q[0], idx_a[0]
  
  # serialize
  #reverse direction questions
  idx_q = np.fliplr(idx_q)
  np.save('questions.npy', idx_q)  
  np.save('answers.npy', idx_a)
  with open('meta.json', 'w') as meta:
    json.dump({'limit': limit, 'w2idx': w2idx, 'idx2w': idx2w}, meta)


  return w2idx, idx2w, idx_q, idx_a


def idxs2str(idxs, idx2w):
  #return " ".join(idx2w[str(c)] for c in idxs if idx2w[str(c)] not in [PAD, BOS, EOS])
  return " ".join(idx2w[str(c)] for c in idxs)

def idxs2chars(idxs, idx2w):
  chars = []
  for idx in idxs:
    char = idx2w[str(idx)]
    if char == EOS:
      break
    if char not in [PAD, EOS]:
      chars.append(char)

  return chars

def str2idxs(str, w2idx):
  filtered = []
  for c in str.split():
    if c not in w2idx:
      c = UNK
    filtered += [w2idx[c]]
  
  return filtered

def bleu_score(predicts, actuals, idx2w):
  
  str_preds =  [idxs2chars(sentence, idx2w) for sentence in predicts]
  str_actual = [idxs2chars(sentence, idx2w) for sentence in actuals]
  
  scores = []
  
  for i in xrange(len(str_preds)):
    score = sentence_bleu(str_actual[i], str_preds[i])
    scores.append(score)

  return np.mean(scores)
