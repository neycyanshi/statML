import numpy as np
import time


def logsumexp(a, dtype=np.float64, axis=None, keepdims=True):
  """Compute ``np.log(np.sum(np.exp(a)))`` in a numerically more stable way.
  Parameters
  ----------
  axis: along which axis to sum

  See Also
  --------
  numpy.logaddexp, numpy.logaddexp2
  """
  a = np.asarray(a, dtype=dtype) # dtype??
  a_max = np.amax(a, axis=axis, keepdims=True)

  if a_max.ndim > 0:
    a_max[~np.isfinite(a_max)] = 0
  elif not np.isfinite(a_max):
    a_max = 0

  # with np.errstate(divide='ignore'): # suppress warnings about log of zero
  out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=keepdims))
  if not keepdims:
    a_max = np.squeeze(a_max, axis=axis)
  out += a_max

  return out


class model(object):
  def __init__(self, K, D ,W):
    """
    :param K: K clusters, num of latent topics
    :param D: D samples, num of docs in corpus(Dataset.docs)
    :param W: vocabulary size
    :return:
    """
    self.K = K
    self.D = D
    self.W = W
    self.PI = np.random.rand(self.K)
    # model parameters {\pi, \mu}, \mu in Categorical distribution.
    self.PI = self.PI/np.sum(self.PI)
    self.MU = np.random.rand(self.K, self.W)
    self.MU = self.MU/np.sum(self.MU, axis=1, keepdims=True)
    self.sim = 0.0

  def train(self, x_, max_iter=100, pz=None, tol=1e-6, eps=1e-30):
    """
    Parameters
    ----------
    x_: training set data, (D,W) word occurrence matrix T
    pz: p(z|x) or q(c_d=k), z means latent topic c_d. (D,K), one instance x on each row, one component z on each column.
    tol: float. stop when change in p(z|x) is less than tol.

    Returns
    -------
    PI: mixing proportions, prob of topic for one doc. (K,)
    MU: component distributions, one component on each row. (K,W)
    """
    # Nd = np.sum(x_, axis=1, keepdims=True) # (D,1)
    if pz is None:
      pz = np.random.dirichlet(np.ones(self.K), self.D) # (D,K) each row sum up to 1, means doc must have one top in K.

    pz_diff = np.finfo(np.float64).max
    i = 0
    while i < max_iter and pz_diff > tol:
      start_time = time.time()
      
      # M step
      self.PI = np.sum(pz, axis=0) / self.D # den is const D, sum of D ones. eq(23)
      tmp = pz.T.dot(x_) + eps # (K,W)
      self.MU = tmp / np.sum(tmp, axis=1, keepdims=True)
      # self.MU = pz.T.dot(x_) / np.sum(pz.T * Nd.T, axis=1, keepdims=True) # (K,W) sum_d[q_dk * sum_w(T_dw)] same
      # self.MU = pz.T.dot(x_) / pz.T.dot(Nd) # (K,W) same as above line
      
      # E step convert prod to sum for numerical stability
      log_pz_num = np.log(self.PI) + x_.dot(np.log(self.MU.T)) # (D,K) denominator of log(p(z|x))
      pz_new = np.exp(log_pz_num - logsumexp(log_pz_num, axis=1)) # (D, K)
      pz_diff = np.amax(np.abs(pz_new - pz))
      pz = pz_new

      i += 1
      iter_duration = time.time() - start_time
      print "Iteration %d, %f seconds." %(i, iter_duration)
      # print "PI: %r, MU: %r" % (self.PI, self.MU)

    print "Training done.\n"
    return self.PI, self.MU, pz

  def output(self, num, dataset):
    """
    :param num: the num of words which rank up to print
    """
    word = [[] for i in range(self.K)]
    f = file('./nips/output.txt', 'w+')
    print "K = %d" %(self.K)
    for i in range(self.K):
      print "The cluster %d\'s ratio is %f, most-frequent %d words are: " % (i+1, self.PI[i], num)
      print>>f, "The cluster %d\'s ratio is %f, most-frequent %d words are: " % (i+1, self.PI[i], num)
      list = []
      for j in range(self.W):
        list.append((self.MU[i,j],j))
      list.sort(key=lambda x:x[0], reverse=True)
      for j in range(num):
        word[i].append(dataset.voc_dict[list[j][1]])
        print dataset.voc_dict[list[j][1]],
        print>>f, dataset.voc_dict[list[j][1]]
      print ""

    self.sim=0.0
    num_total = 0.0
    for i in range(self.K - 1):
      for j in range(i+1,self.K):
        cnt = 0.0
        for k in range(5):
          for l in range(5):
            if word[i][k] is word[j][l]:
              cnt += 1.0
        # print cnt
        num_total += 1.0
        self.sim += cnt / 5.0
    self.sim /= num_total
    print "Average self.similarity is %f\n" %(self.sim)
    print >>f,"Average self.similarity is %f\n" %(self.sim)

