# Intrinsic fluctuations of reinforcement learning promote cooperation Code Repository
This repository contains all code to reproduce the results of [this work](https://arxiv.org/abs/2209.01013).

[![DOI](https://zenodo.org/badge/562998950.svg)](https://zenodo.org/badge/latestdoi/562998950)


## 1 | Stability
In the folder `1_Stability`, you find the Mathematica code for determining the absorbing states and basins of attraction of the mutual best-response dynamics.

The Mathematica file `Expected_SARSA_Bellman_Solutions.nb` was written using Mathematica 13 and will require an installation of Mathematica to be executed (see [installation requirements and instructions](https://reference.wolfram.com/language/tutorial/InstallingMathematica.html)). 

It contains three sections, the first for calculating possible best-response relations, the second collecting the results of such calculations in the case of the iterated prisoner's dilemma as considered in the article and the third plotting the best-response graph given a choice of model parameters.

## 2 | Learnability 
In the folder `2_Learnability`, you find the python code for the deterministic learning dynamics. 

The notebook `plots.ipynb` walks you through producing the plots of the paper. If you want to simulate the learning dynamics yourself, you need the [JAX package](https://github.com/google/jax).


## 3 | Stochasticity
In the folder `3_Stochasticity`, you find the python code for the Sample-Batch Expected SARSA algorithm.

The file `Batched_ExpectedSARSA_IPD_Numba.py` contains the simulation for the Sample-Batch Expected SARSA algorithm. It requires an [installation of numba](https://numba.pydata.org/numba-doc/latest/user/installing.html) and makes use of functions defined in the file `IPD_functions.py`. 

The parameter choices can be set in the "Simulation Parameters" section with Rt corresponding to the paremeter T, Rs corresponding to the parameter S, deltaQ corresponding to the discount factor, constanteps corresponding to the exploration rate, alpha corresponding to the learning rate, BatchK corresponding to the batch size, NBatches corresponding to the number of batches and NSamples corresponding to the number of sample trajectories. 

Running the simulation outputs four files: AllD.txt containing the fraction of trajectories in the AllD strategy pair as a function of time, GT.txt containing the fraction of trajectories in the GT strategy pair as a function of time, WSLS.txt containing the fraction of trajectories in the WSLS strategy pair as a function of time and ColLevel.txt containing the average of the fraction of times the algorithms cooperated in the current batch.


---
Feel free to reach out, or open an issue, if you have any questions.

Contact: wolfram.barfuss@gmail.com | j.m.meylahn@utwente.nl
