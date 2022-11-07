import numpy as np
import numba
from numba import njit
import math
import random
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import minimize
from scipy.optimize import fsolve
from IPD_functions import *
import time
from array import array

#Simulation paramters
NSamples = 1000
NBatches = 400
BatchK = 3000
FinalT = NBatches*BatchK
M = 2
Rp=0.
Rr=1.
Rs=-0.5
Rt=1.5
xi = 0.0
option = 'uniform' #Choosing the type of Q-matrix initialization. Options are std_normal, uniform, constant or Calvano.
alpha = 0.15
deltaQ = 0.95
constanteps = 0.1

#Initialize environment
actionspaceinitial = np.array([[0, 1], [0, 1]], dtype=float)
statespaceinitial = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
priceinterval = np.array([0, 1], dtype=float)
WSLSstratinitial = np.array([1, 0, 0, 1, 1, 0, 0, 1], dtype=float)
AllDstratinitial = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
GTstratinitial = np.array([0, 0, 0, 1, 0, 0, 0, 1], dtype=float)
payoffs1 = np.array([[Rp, Rt], [Rs, Rr]], dtype=float)
payoffs2 = np.array([[Rp, Rs], [Rt, Rr]], dtype=float)

qmatrix1initial = np.array(Qinitialization(option, M, 0, deltaQ, payoffs1), dtype=float)
qmatrix2initial = np.array(Qinitialization(option, M, 1, deltaQ, payoffs2), dtype=float)

#qmatrix1initial = np.array([[1, 0],[1, 0],[1, 0],[1, 0]], dtype=float)
#qmatrix2initial = np.array([[1, 0],[1, 0],[1, 0],[1, 0]], dtype=float)


#Open file for collecting data
file1 = open('AllD.txt', 'w')
file2 = open('GT.txt', 'w')
file3 = open('WSLS.txt', 'w')
file4 = open('ColLevel.txt', 'w')

#Start timer
start_time = time.time()

#The function performing the simulation
def QvsQsimulation(FinalT, qmatrix1initial, qmatrix2initial, actionspaceinitial, statespaceinitial, payoffs1, payoffs2, deltaQ):
    revenuesamp1 = np.zeros(NSamples)
    revenuesamp2 = np.zeros(NSamples)
    Collist = np.zeros(int(NBatches)+1)
    ADlist= np.zeros(int(NBatches)+1)
    WSLSlist=np.zeros(int(NBatches)+1)
    GTlist=np.zeros(int(NBatches)+1)
    actionspacetemp = actionspaceinitial
    statespacetemp = statespaceinitial
    zerosM =list(np.zeros(M))
    actionspace = [zerosM, zerosM]
    statespace = []
    WSLSstrat = []
    GTstrat = []
    AllDstrat = []

    #Absorbing state strategies
    for listitA in range(M):
        actionspace[0][listitA] = round(actionspacetemp[0][listitA], 5)
        actionspace[1][listitA] = round(actionspacetemp[1][listitA], 5)
    for listitB in range(M*M):
        statespace.append([round(statespacetemp[listitB][0], 5), round(statespacetemp[listitB][1], 5)])
    for listitG in range(2*M*M):
        WSLSstrat.append(WSLSstratinitial[listitG])
        GTstrat.append(GTstratinitial[listitG])
        AllDstrat.append(AllDstratinitial[listitG])

    #Loop over Samples
    for n in range(NSamples):
        #Initial Q-matrix
        if option == 'constant' or option == 'Calvano' or option == 'constant_scaled':
            qmatrix1act=np.zeros((M*M, M))
            qmatrix2act=np.zeros((M*M, M))
            qmatrix1value=np.zeros((M*M, M))
            qmatrix2value=np.zeros((M*M, M))
            for listitD in range(M*M):
                for listitE in range(M):
                    qmatrix1act[listitD][listitE] = qmatrix1initial[listitD][listitE]
                    qmatrix2act[listitD][listitE] = qmatrix2initial[listitD][listitE]
                    qmatrix1value[listitD][listitE] = qmatrix1initial[listitD][listitE]
                    qmatrix2value[listitD][listitE] = qmatrix2initial[listitD][listitE]
        elif option == 'std_normal':
            qmatrix1act=np.zeros((M*M, M))
            qmatrix2act=np.zeros((M*M, M))
            qmatrix1value=np.zeros((M*M, M))
            qmatrix2value=np.zeros((M*M, M))
            for listitD in range(M*M):
                for listitE in range(M):
                    qmatrix1act[listitD][listitE] = np.random.normal(0, 1)
                    qmatrix2act[listitD][listitE] = np.random.normal(0, 1)
                    qmatrix1value[listitD][listitE] = qmatrix1act[listitD][listitE]
                    qmatrix2value[listitD][listitE] = qmatrix2act[listitD][listitE]

        elif option == 'uniform':
            qmatrix1act=np.zeros((M*M, M))
            qmatrix2act=np.zeros((M*M, M))
            qmatrix1value=np.zeros((M*M, M))
            qmatrix2value=np.zeros((M*M, M))
            for listitD in range(M*M):
                for listitE in range(M):
                    qmatrix1act[listitD][listitE] = np.random.uniform(0, 1)
                    qmatrix2act[listitD][listitE] = np.random.uniform(0, 1)
                    qmatrix1value[listitD][listitE] = qmatrix1act[listitD][listitE]
                    qmatrix2value[listitD][listitE] = qmatrix2act[listitD][listitE]

        #Calculate current strategy
        strategypairvec=np.zeros(2*M*M)
        convergedstrat = np.zeros(2*M*M)
        for l in range(M*M):
            maxindex1 = np.argmax(qmatrix1act[l])
            maxindex2 = np.argmax(qmatrix2act[l])
            strategypairvec[l] =  maxindex1
            convergedstrat[l] = maxindex1
            strategypairvec[l+(M*M)] =  maxindex2
            convergedstrat[l+(M*M)] = maxindex2

        #Record Initial strategies
        matchWSLS = 1
        matchAllD = 1
        matchGT = 1
        for listitF in range(M*M):
            if int(convergedstrat[listitF]) != int(WSLSstrat[listitF]) or int(convergedstrat[listitF+(M*M)]) != int(WSLSstrat[listitF+(M*M)]):
                matchWSLS = 0
            if int(convergedstrat[listitF]) != int(GTstrat[listitF]) or int(convergedstrat[listitF+(M*M)]) != int(GTstrat[listitF+(M*M)]):
                matchGT = 0
            if int(convergedstrat[listitF]) != int(AllDstrat[listitF]) or int(convergedstrat[listitF+(M*M)]) != int(AllDstrat[listitF+(M*M)]):
                matchAllD = 0

        #Record if it is one of the absorbing states
        if matchWSLS == 1:
            WSLSlist[0] = WSLSlist[0] + 1./NSamples   
        elif matchGT == 1:
            GTlist[0] =  GTlist[0] + 1./NSamples
        elif matchAllD == 1:
            ADlist[0] =  ADlist[0] + 1./NSamples 

        #Pricing history
        pricetraj1 = np.zeros(FinalT)
        pricetraj2 = np.zeros(FinalT)

        #Initialize time and convergence status (only for normal (Calvano) delta)
        t = int(0)
        nbatch = int(0)

        #Random initial price/state
        randomactionindex1 = np.random.randint(0, len(actionspace[0]))
        randomactionindex2 = np.random.randint(0, len(actionspace[1]))
        pricetraj1[0] = float(actionspace[0][randomactionindex1])
        pricetraj2[0] = float(actionspace[1][randomactionindex2])
        s = [pricetraj1[0], pricetraj2[0]]

        
        #Initialize average reward of LastInt last periods of the simulation to zero
        totalrevenue1 = 0.0    
        totalrevenue2 = 0.0
        strategypairvec=np.zeros(2*M*M) 

        #Loop for Normal (Calvano) Delta  
        while nbatch < NBatches:

            #Transfer act to value matrix
            if nbatch > 1:
                for listitM in range(M*M):
                    for listitN in range(M):
                        qmatrix1value[listitM][listitN] = qmatrix1act[listitM][listitN]
                        qmatrix2value[listitM][listitN] = qmatrix2act[listitM][listitN]

            #Initialize batch data structures
            q1reward = np.zeros((M*M, M))
            q2reward = np.zeros((M*M, M))
            q1next = np.zeros((M*M, M))
            q2next = np.zeros((M*M, M))
            q1count = np.zeros((M*M, M))
            q2count = np.zeros((M*M, M))
            Tsas1 = np.zeros((M*M, M, M*M))
            Tsas2 = np.zeros((M*M, M, M*M))
            nbatch = nbatch + 1
            kbatch = int(0)
            collusioncount = 0.
            while kbatch < BatchK:
                t = t+1
                kbatch = kbatch + 1

                #Action-selection mechanism    
                random1 = np.random.uniform(0.0, 1.0)
                random2 = np.random.uniform(0.0, 1.0)
                sindex = statespace.index(s)
                if random1<constanteps:
                    randomactionindex1 = np.random.randint(0, len(actionspace[0]))
                    pricetraj1[t] = actionspace[0][randomactionindex1]
                else:
                    maxpos1 = np.argmax(qmatrix1act[sindex])
                    pricetraj1[t] = actionspace[0][maxpos1]

                if random2<constanteps:
                    randomactionindex2 = np.random.randint(0, len(actionspace[1]))
                    pricetraj2[t] = actionspace[1][randomactionindex2]
                else:
                    maxpos2 = np.argmax(qmatrix2act[sindex])
                    pricetraj2[t] = actionspace[1][maxpos2]

                #Update state
                s[0]  =  pricetraj1[t]
                s[1]  =  pricetraj2[t]
                s1index = actionspace[0].index(s[0])
                s2index = actionspace[1].index(s[1])

                #Batch data collection
                q1count[sindex][s1index] = q1count[sindex][s1index] + 1.
                q2count[sindex][s2index] = q2count[sindex][s2index] + 1.
                q1reward[sindex][s1index] = q1reward[sindex][s1index] + payoffs1[s1index][s2index]
                q2reward[sindex][s2index] = q2reward[sindex][s2index] + payoffs2[s1index][s2index]
                Tsas1[sindex][s1index][statespace.index(s)] = Tsas1[sindex][s1index][statespace.index(s)] + 1.
                Tsas2[sindex][s2index][statespace.index(s)] = Tsas2[sindex][s2index][statespace.index(s)] + 1.
                if q1count[sindex][s1index] > 1:
                    alphaaux = 1./(q1count[sindex][s1index]+1.)
                    qmatrix1value[sindex][s1index] = (1.-alphaaux)*qmatrix1value[sindex][s1index] + alphaaux*(payoffs1[s1index][s2index] + deltaQ*((1-constanteps/2)*max(qmatrix1value[statespace.index(s)])+(constanteps/2)*min(qmatrix1value[statespace.index(s)])))
                else:
                    alphaaux = 1.
                    qmatrix1value[sindex][s1index] = (1.-alphaaux)*qmatrix1value[sindex][s1index] + alphaaux*(payoffs1[s1index][s2index]+deltaQ*((1-constanteps/2)*max(qmatrix1value[statespace.index(s)])+(constanteps/2)*min(qmatrix1value[statespace.index(s)])))
                if q2count[sindex][s1index] > 1:
                    alphaaux = 1./(q2count[sindex][s2index]+1.)
                    qmatrix2value[sindex][s2index] = (1.-alphaaux)*qmatrix2value[sindex][s2index] + alphaaux*(payoffs2[s1index][s2index] + deltaQ*((1-constanteps/2)*max(qmatrix2value[statespace.index(s)])+(constanteps/2)*min(qmatrix2value[statespace.index(s)])))      
                else:
                    alphaaux = 1.
                    qmatrix2value[sindex][s2index] = (1.-alphaaux)*qmatrix2value[sindex][s2index] + alphaaux*(payoffs2[s1index][s2index]+ deltaQ*((1-constanteps/2)*max(qmatrix2value[statespace.index(s)])+(constanteps/2)*min(qmatrix2value[statespace.index(s)])))      
                if pricetraj1[t]==1 and pricetraj2[t]==1:
                    collusioncount = collusioncount + 1./BatchK
       
            #Post batch Q-value update
            for listitJ in range(M*M):
                for listitK in range(M):
                    if q1count[listitJ][listitK] == 0.:
                        q1reward[listitJ][listitK] = qmatrix1act[listitJ][listitK]
                    if q2count[listitJ][listitK] == 0.:
                        q2reward[listitJ][listitK] = qmatrix2act[listitJ][listitK]
                    for listitL in range(M*M):
                        q1next[listitJ][listitK] = q1next[listitJ][listitK] + (Tsas1[listitJ][listitK][listitL]/max([q1count[listitJ][listitK], 1.]))*((1-constanteps/2)*max(qmatrix1value[listitL])+(constanteps/2)*min(qmatrix1value[listitL]))
                        q2next[listitJ][listitK] = q2next[listitJ][listitK] + (Tsas2[listitJ][listitK][listitL]/max([q2count[listitJ][listitK], 1.]))*((1-constanteps/2)*max(qmatrix2value[listitL])+(constanteps/2)*min(qmatrix2value[listitL]))

                    qmatrix1act[listitJ][listitK] = (1.-alpha)*qmatrix1act[listitJ][listitK] +alpha*(q1reward[listitJ][listitK]/max([q1count[listitJ][listitK], 1])+deltaQ*q1next[listitJ][listitK]) 
                    qmatrix2act[listitJ][listitK] = (1.-alpha)*qmatrix2act[listitJ][listitK] +alpha*(q2reward[listitJ][listitK]/max([q2count[listitJ][listitK], 1])+deltaQ*q2next[listitJ][listitK]) 

            #Calculate current strategy
            convergedstrat = np.zeros(2*M*M)
            for l in range(M*M):
                maxindex1 = np.argmax(qmatrix1act[l])
                maxindex2 = np.argmax(qmatrix2act[l])
                strategypairvec[l] =  maxindex1
                convergedstrat[l] = maxindex1
                strategypairvec[l+(M*M)] =  maxindex2
                convergedstrat[l+(M*M)] = maxindex2

            #Check if strategy matches absorbing state
            matchWSLS = 1
            matchAllD = 1
            matchGT = 1
            for listitF in range(M*M):
                if int(convergedstrat[listitF]) != int(WSLSstrat[listitF]) or int(convergedstrat[listitF+(M*M)]) != int(WSLSstrat[listitF+(M*M)]):
                    matchWSLS = 0
                if int(convergedstrat[listitF]) != int(GTstrat[listitF]) or int(convergedstrat[listitF+(M*M)]) != int(GTstrat[listitF+(M*M)]):
                    matchGT = 0
                if int(convergedstrat[listitF]) != int(AllDstrat[listitF]) or int(convergedstrat[listitF+(M*M)]) != int(AllDstrat[listitF+(M*M)]):
                    matchAllD = 0
            #Record if it is one of the absorbing states
            if matchWSLS == 1:
                WSLSlist[int(nbatch)] = WSLSlist[int(nbatch)] + 1./NSamples   
            elif matchGT == 1:
                GTlist[int(nbatch)] =  GTlist[int(nbatch)] + 1./NSamples
            elif matchAllD == 1:
                ADlist[int(nbatch)] =  ADlist[int(nbatch)] + 1./NSamples 

            Collist[int(nbatch)] = Collist[int(nbatch)] + collusioncount/NSamples


    return WSLSlist, GTlist, ADlist, Collist

simulate_numba = njit(QvsQsimulation)


WSLSlist, GTlist, ADlist, Collist = simulate_numba(FinalT, qmatrix1initial, qmatrix2initial, actionspaceinitial, statespaceinitial, payoffs1, payoffs2, deltaQ)
for listitH in range(int(NBatches)+1):
    file1.write(str(ADlist[listitH]) + "\n")
    file2.write(str(GTlist[listitH]) + "\n")
    file3.write(str(WSLSlist[listitH]) + "\n")
    file4.write(str(Collist[listitH]) + "\n")

progtime = time.time() - start_time
if progtime < 60:
    print("My program took", round(progtime, 2), "seconds to run.")
if progtime > 60:
    print("My program took", round(progtime/60, 2), "minutes to run.")

file1.close()
file2.close()
file3.close()
file4.close()
                   
