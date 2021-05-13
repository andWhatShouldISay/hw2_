import torch
import numpy as np
import scipy
import pandas as pd
from scipy.sparse import coo_matrix
from collections import Counter

train = pd.read_csv("train.csv")

ratings = train['rating'].values
userIds = train['userId'].values
itemIds = train['movieId'].values

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

userIds = np.array(list(map(lambda x: compr_u[x], userIds)))
itemIds = np.array(list(map(lambda x: compr_i[x], itemIds)))
total = len(ratings)

R = coo_matrix((ratings, (userIds, itemIds)), shape=(n_users, n_items))

rateds_ = R.nonzero()
rateds = set()

for i in range(len(rateds_[0])):
  rateds.add((rateds_[0][i],rateds_[1][i]))

k = 5 # hyperparameter
la = 1e-5 # hyperparameter

print(total)

P = torch.rand(n_users, k) * 2
Q = torch.rand(n_items, k) * 2
bu = torch.rand(n_users) * 2
bi = torch.rand(n_items) * 2
nu = torch.rand(1) * 10 - 5

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

def mse(P,Q,R,bu,bi,nu,userIds=userIds,itemIds=itemIds):
  m = that_stupid_thing(P,Q,R,bu,bi,nu,userIds=userIds,itemIds=itemIds)
  return np.sum(m.power(2))/total


def loss(P,Q,R,bu,bi,nu,la,userIds=userIds,itemIds=itemIds):
  return mse(P,Q,R,bu,bi,nu,userIds=userIds,itemIds=itemIds) + la*(LA.norm(P)**2 + LA.norm(Q)**2)

print(loss(P,Q,R,bu,bi,nu,la))

lr = 1e-4
def updateP(P,Q,R,bu,bi,nu,la):
  m = that_stupid_thing(P,Q,R,bu,bi,nu)
  grad = la * P + m @ Q 
  return [P - lr * grad, grad]


def updateQ(P,Q,R,bu,bi,nu,la):
  m = that_stupid_thing(P,Q,R,bu,bi,nu)
  grad = la * Q + m.T @ P 
  return [Q - lr * grad, grad]  

def update_bubinu(P,Q,R,bu,bi,nu,la):
  m = that_stupid_thing(P,Q,R,bu,bi,nu)
  grad_bu = np.divide(np.sum(m,axis = 1),rated_u)
  grad_bi = np.divide(np.sum(m,axis = 0).T,rated_i)
  grad_nu = np.sum(m)/total
  return [[bu - lr*np.ravel(grad_bu), bi - lr*np.ravel(grad_bi), nu-lr*grad_nu], [grad_bu,grad_bi,grad_nu]]

print("loss: ",loss(P,Q,R,bu,bi,nu,la))
while True:
  [P, gradP] = updateP(P,Q,R,bu,bi,nu,la)
  [Q, gradQ] = updateQ(P,Q,R,bu,bi,nu,la)
  [[bu,bi,nu], [grad1, grad2, grad3]] = update_bubinu(P,Q,R,bu,bi,nu,la)
  print("loss: ", loss(P,Q,R,bu,bi,nu,la))
  print("mse: ", mse(P,Q,R,bu,bi,nu))
  g1 = LA.norm(gradP)
  g2 = LA.norm(gradQ)
  print("gradient: ", g1, g2, '\n')
  if g1 <= 400.0 or g2 <= 400.0:
    break
    
np.save("p.npy",P)
np.save("q.npy",Q)
np.save("bu.npy",bu)
np.save("bi.npy",bi)
np.save("nu.npy",nu)

def recommend(u):
  u = compr_u[u]
  R_hat = P[u] @ Q.T
  rec = []
  for i in range(n_items):
    if not (u,i) in rateds:
      rec.append(((R_hat[i] + bu[u] + bi[i] + nu)[0],compr_it[i]))
  rec.sort(reverse = True)
  print(rec[0:5])

recommend(1)


