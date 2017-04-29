import numpy as np


class Dataset(object):
  def __init__(self, voc_path, corpus_path):
    self.num_docs = 0
    self.voc_size = 0
    self.docs = None
    self.voc_dict = {}
    self.voc_path = voc_path
    self.corpus_path = corpus_path
    self.create_vocDict(voc_path)
    self.create_trainset(corpus_path)

  def create_vocDict(self, voc_path):
    with open(voc_path) as voc_file:
      for line in voc_file.readlines():
        word_idx, word, freq = line.split('\t')
        self.voc_dict[int(word_idx)] = word
        self.voc_size += 1
    print('vocabulary dict read done.')

  def create_trainset(self, corpus_path):
    with open(corpus_path) as docs_file:
      docs_raw = docs_file.readlines()
      self.num_docs = len(docs_raw)
      self.docs = np.zeros((self.num_docs, self.voc_size), dtype=np.float32) # (D, W) float64??
      for doc_idx, line in enumerate(docs_raw):
        word_bag = line.strip().split('\t')[1].split(' ')
        for word in word_bag:
          word_idx, word_count = word.split(':')
          self.docs[doc_idx, int(word_idx)] = int(word_count)
    print('corpus read done.')
