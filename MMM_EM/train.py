import numpy as np
import Input
import model


voc_file = "./nips/nips.vocab"
corpus_file = "./nips/nips.libsvm"

cluster_num_list = [5,10,20,30]
best_sim = 1.0
best_id = 0

for i in range(len(cluster_num_list)):
  cluster_num = cluster_num_list[i]
  Dataset = Input.Dataset(voc_path=voc_file, corpus_path=corpus_file)

  # multinomial mixture model
  mmm = model.model(cluster_num, Dataset.num_docs, Dataset.voc_size)
  mmm.train(x_=Dataset.docs, max_iter=20)
  mmm.output(5, dataset=Dataset)

  if mmm.sim < best_sim:
    best_sim = mmm.sim
    best_id = i

print "The best K is: %d " %(cluster_num_list[best_id])
