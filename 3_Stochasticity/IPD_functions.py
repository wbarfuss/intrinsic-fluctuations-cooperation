import numpy as np
import math
import random
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import minimize
from scipy.optimize import fsolve

#Environment functions

def rewardIPD(s, i, Rt, Rr, Rp, Rs):
    if s==0:
        rewards = [Rp, Rp]
    if s==1:
        rewards = [Rt, Rs]
    if s==2:
        rewards = [Rs, Rt]
    if s==3:
        rewards = [Rr, Rr]        
    return rewards[i]

def payoffmatrix(m, actionspace, Rt, Rr, Rp, Rs):
    payoffmat = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            if m == 0:
                payoffmat[i][j] = float(rewardIPD([i, j], m, Rt, Rr, Rp, Rs))
            if m == 1:
                payoffmat[i][j] = float(rewardIPD([i, j], m, Rt, Rr, Rp, Rs))
    return payoffmat


#Q-learning functions

def epsilon(t, beta, e0):
    return e0*math.exp(-t*beta)

def steppedepsilon(t, beta):
    if t < 5000:
        eps = 0.8
    if t > 4999 and t < 10000:
        eps = 0.4 
    if t > 9999:
        eps = 0.01
    return eps

def Qinitialization(option, M, m, delta, payoffs):
    if option == 'std_normal':
        return np.random.normal(0, 1, size=(M*M, M))
    elif option == 'uniform':
        return np.random.uniform(0, 1, size=(M*M, M))
    elif option == 'constant':
        return np.full((M*M,M), 0.5)
    elif option == 'Calvano':
        qmatrixtemp = np.zeros((M*M, M))
        if m == 0:
            for i in range(M):
                aveactionpay = 0
                for j in range(M):
                    aveactionpay += payoffs[i][j]
                for l in range(M*M):
                    qmatrixtemp[l][i] = aveactionpay/((1-delta)*(M)) 
            return qmatrixtemp
        if m == 1:
            for i in range(M):
                aveactionpay = 0
                for j in range(M):
                    aveactionpay += payoffs[j][i]
                for l in range(M*M):
                    qmatrixtemp[l][i] = aveactionpay/((1-delta)*(M))      
            return qmatrixtemp
    else:
        raise ValueError("Invalid initialization option.")

#Exp3 functions    

def Exp3StratDist(Exp3, M, eta):
    dist = [0]*M
    normalize = 0.
    for l in range(M):
        normalize = normalize + math.exp(eta*Exp3[l])
    for m in range(M):
        dist[m] = math.exp(eta*Exp3[m])/normalize
    return dist

def optimal_eta(M, deltaExp):
    return math.sqrt(math.log(float(M))*(1-deltaExp**2)/(float(M)*(deltaExp**2)))
    

#Evaluation functions

def strategypair2D(qmatrix1, qmatrix2):
    if qmatrix1[0][0]>qmatrix1[0][1] and qmatrix2[0][0]>qmatrix2[0][1] and qmatrix1[1][0]>qmatrix1[1][1] and qmatrix2[1][0]>qmatrix2[1][1] and qmatrix1[2][0]>qmatrix1[2][1] and qmatrix2[2][0]>qmatrix2[2][1] and qmatrix1[3][0]>qmatrix1[3][1] and qmatrix2[3][0]>qmatrix2[3][1]:
        return 0
    if qmatrix1[0][0]>qmatrix1[0][1] and qmatrix2[0][0]>qmatrix2[0][1] and qmatrix1[1][0]>qmatrix1[1][1] and qmatrix2[1][0]>qmatrix2[1][1] and qmatrix1[2][0]>qmatrix1[2][1] and qmatrix2[2][0]>qmatrix2[2][1] and qmatrix1[3][0]<qmatrix1[3][1] and qmatrix2[3][0]<qmatrix2[3][1]:
        return 1
    if qmatrix1[0][0]<qmatrix1[0][1] and qmatrix2[0][0]<qmatrix2[0][1] and qmatrix1[1][0]>qmatrix1[1][1] and qmatrix2[1][0]>qmatrix2[1][1] and qmatrix1[2][0]>qmatrix1[2][1] and qmatrix2[2][0]>qmatrix2[2][1] and qmatrix1[3][0]<qmatrix1[3][1] and qmatrix2[3][0]<qmatrix2[3][1]:
        return 2
    else:
        return 3

def strategypairabsorbstate(M, qmatrix1, qmatrix2, actionspace, statespace):
    graph = [0]*(M*M)
    absorbstate = []
    for n in range(M*M):
        maxindex1 = np.where(qmatrix1[n] == max(qmatrix1[n]))
        maxindex2 = np.where(qmatrix2[n] == max(qmatrix2[n]))
        graph[n] =  statespace.index([actionspace[0][maxindex1[0][0]], actionspace[1][maxindex2[0][0]]])   
        if graph[n] == n:
            absorbstate.append(n)
    return absorbstate

def nashabsorbstate(M, statespace, actionspace, a1, a2, mu, c1, c2):
    nprice1temp = effectiveNash(0, M, actionspace, a1, a2, mu, c1, c2)[1]
    nprice2temp = effectiveNash(1, M, actionspace, a1, a2, mu, c1, c2)[1]
    return statespace.index([nprice1temp, nprice2temp])

def collusivestratpair(M, qmatrix1, qmatrix2, nashstate, actionspace, statespace):
    if len(strategypairabsorbstate(M, qmatrix1, qmatrix2, actionspace, statespace))>0 and strategypairabsorbstate(M, qmatrix1, qmatrix2, actionspace, statespace)[0]>nashstate:
        return 2
    if len(strategypairabsorbstate(M, qmatrix1, qmatrix2, actionspace, statespace))>0 and strategypairabsorbstate(M, qmatrix1, qmatrix2, actionspace, statespace)[0]==nashstate:
        return 1
    else:
        return 0

def strategybit(M, qmatrix1, qmatrix2):
    bit = [0]*(2*M*M)
    for u in range(M*M):
        maxQpos1 = np.where(qmatrix1[u] == max(qmatrix1[u]))
        maxQpos2 = np.where(qmatrix2[u] == max(qmatrix2[u]))
        bit[u] = maxQpos1[0][0]
        bit[u+(M*M)] = maxQpos2[0][0]
    return bit

