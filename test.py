import torch
import numpy as np
import scipy
import pandas as pd
from scipy.sparse import coo_matrix
from collections import Counter

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

ratings = train['rating'].values
userIds = train['userId'].values
itemIds = train['movieId'].values
ratings_test = test['rating'].values
userIds_test = test['userId'].values
itemIds_test = test['movieId'].values

cnt_u = Counter(userIds)
cnt_i = Counter(itemIds)

def compress(cnt):
  d = {}
  dt = {}
  z = 0
  for x in cnt:
    d[x] = z
    dt[z] = x
    z += 1
  return d, dt


[compr_u, compr_ut] = compress(cnt_u)
[compr_i, compr_it] = compress(cnt_i)

rated_u = np.array([cnt_u[compr_ut[i]] for i in compr_ut]).reshape(-1,1)
rated_i = np.array([cnt_i[compr_it[i]] for i in compr_it]).reshape(-1,1)

n_users = len(compr_u)
n_items = len(compr_i)

print(n_users, n_items)

userIds_test = np.array(list(map(lambda x: compr_u[x], userIds_test)))
itemIds_test = np.array(list(map(lambda x: compr_i[x], itemIds_test)))
total_test = len(ratings_test)
total = len(ratings)

R_test = coo_matrix((ratings_test, (userIds_test, itemIds_test)), shape=(n_users, n_items))


la = 1e-5 # hyperparameter

P = torch.from_numpy(np.load('p.npy'))
Q = torch.from_numpy(np.load('q.npy'))
bu = torch.from_numpy(np.load('bu.npy'))
bi = torch.from_numpy(np.load('bi.npy'))
nu = torch.from_numpy(np.load('nu.npy'))

def that_stupid_thing(P,Q,R,bu,bi,nu,userIds=userIds,itemIds=itemIds):
  nonzero = R.nonzero()
  P_tau = P[nonzero[0],:]
  Q_tau = Q[nonzero[1],:]
  bu_tau = bu[nonzero[0]]
  bi_tau = bi[nonzero[1]]
  R_hat_values = torch.sum(np.multiply(P_tau,Q_tau), dim = 1) + bu_tau + bi_tau + nu
  R_hat = coo_matrix((R_hat_values, (userIds, itemIds)), shape=(n_users, n_items))
  return R_hat - R

from numpy import linalg as LA

def mse(P,Q,R,bu,bi,nu,userIds=userIds,itemIds=itemIds,total=total):
  m = that_stupid_thing(P,Q,R,bu,bi,nu,userIds=userIds,itemIds=itemIds)
  return np.sum(m.power(2))/total


def loss(P,Q,R,bu,bi,nu,la,userIds=userIds,itemIds=itemIds,total=total):
  return mse(P,Q,R,bu,bi,nu,userIds=userIds,itemIds=itemIds,total=total) + la*(LA.norm(P)**2 + LA.norm(Q)**2)

print(loss(P,Q,R_test,bu,bi,nu,la,userIds_test,itemIds_test,total_test), mse(P,Q,R_test,bu,bi,nu,userIds_test,itemIds_test,total_test))
