import os
import sys
from pathlib import Path
base_dir = Path(os.path.abspath('')).resolve().parent.parent.parent
sys.path.append(str(base_dir))

# from LearningDynamics.Environments import BaseEnvironment
from learningdynamics.BaseEnvironment import BaseEnvironment
import numpy as np

class IteratedPrisonersDilemma(BaseEnvironment):
    """
    Symmetric 2-agent 2-action matrix games with history-1 embedding.
    """ 

    def __init__(self, R, T, S, P):
        """
        Specified with the Prisoner's Dilemma nomenclature
        
        R = reward of mutual cooperation
        T = temptation of unilateral defection
        S = sucker's payoff of unilateral cooperation
        P = punishment of mutual defection
        """
        self.N = 2
        self.M = 2
        self.Z = 4

        self.Re = R
        self.Te = T
        self.Su = S    
        self.Pu = P

        # --
        self.state = 1 # inital state
        super().__init__()

    def actions(self):
        """The action sets"""
        return [['c', 'd'] for _ in range(self.N)]

    def states(self):
        """The state sets"""
        return ['CC', 'CD', 'DC', 'DD']

    def TransitionTensor(self):
        """The transition model in tensor form T[s,ja,s_]"""
        dim = np.concatenate(([self.Z],
                              [self.M for _ in range(self.N)],
                              [self.Z]))
        Tsas = np.ones(dim) * (-1)

        for index, _ in np.ndenumerate(Tsas):
            Tsas[index] = self._transition_probability(index[0],
                                                       index[1:-1],
                                                       index[-1])
        return Tsas
    
    def _transition_probability(self, s, jA, sprim):
        
        if (jA[0], jA[1], sprim) == (0, 0, 0):
            return 1.0
        elif (jA[0], jA[1], sprim) == (0, 1, 1):
            return 1.0      
        elif (jA[0], jA[1], sprim) == (1, 0, 2):
            return 1.0      
        elif (jA[0], jA[1], sprim) == (1, 1, 3):
            return 1.0
        else:
            return 0.0

    def RewardTensor(self):
        """The reward model in tensor form R[i,s,ja,s_]"""
        dim = np.concatenate(([self.N],
                              [self.Z],
                              [self.M for _ in range(self.N)],
                              [self.Z]))
        Risas = np.zeros(dim)

        for index, _ in np.ndenumerate(Risas):
            Risas[index] = self._reward(index[0], index[1], index[2:-1],
                                        index[-1])
        return Risas

    def _reward(self, i, s, jA, sprim):

        if (jA[0], jA[1]) == (0, 0):
            return self.Re
        elif (jA[0], jA[1]) == (1, 1):
            return self.Pu
        elif (jA[0], jA[1]) == (0, 1):
            if i == 0:
                return self.Su
            elif i == 1:
                return self.Te
        elif (jA[0], jA[1]) == (1, 0):
            if i == 0:
                return self.Te
            elif i == 1:
                return self.Su
    
    def id(self):
        """
        Returns id string of environment
        """
        # Default
        id = f"{self.__class__.__name__}_"+\
            f"{self.Te}_{self.Re}_{self.Pu}_{self.Su}"
        return id
        