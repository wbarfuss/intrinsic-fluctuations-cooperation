import jax
import numpy as np
import itertools as it

import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial

# =============================================================================
#   Base
# =============================================================================   
class memomeanfield_learning_base(object):
    """
    Base class for deterministic policy-average memory-mean-field
    independent (multi-agent) temporal-difference reinforcement learning.

    To be used as a base for both, value and policy dynamics.
    """
    
    def __init__(self, TransitionTensor, RewardTensor, DiscountFactors,
                 use_prefactor=False,
                 opteinsum=True,
                 **kwargs):
        """
        Parameters
        ----------
        TransitionTensor : transition model of the environment
        RewardTensor : reward model of the environment
        DiscountFactors : the agents' discount factors
        use_prefactor : use the 1-DiscountFactor prefactor (default: False)
        opteinsum : keyword argument to optimize einsum methods (default: True)
        """
        R = jnp.array(RewardTensor)
        T = jnp.array(TransitionTensor)
    
        # number of agents
        N = R.shape[0]  
        assert len(T.shape[1:-1]) == N, "Inconsistent number of agents"
        assert len(R.shape[2:-1]) == N, "Inconsistent number of agents"
        
        # number of actions for each agent        
        M = T.shape[1] 
        assert np.allclose(T.shape[1:-1], M), 'Inconsisten number of actions'
        assert np.allclose(R.shape[2:-1], M), 'Inconsisten number of actions'
        
        # number of states
        Z = T.shape[0] 
        assert T.shape[-1] == Z, 'Inconsisten number of states'
        assert R.shape[-1] == Z, 'Inconsisten number of states'
        assert R.shape[1] == Z, 'Inconsisten number of states'
        
        self.R, self.T, self.N, self.M, self.Z = R, T, N, M, Z
        
        # discount factors
        self.gamma = make_variable_vector(DiscountFactors, N)

        # use (1-DiscountFactor) prefactor to have values on scale of rewards
        self.pre = 1 - self.gamma if use_prefactor else jnp.ones(N)        
        self.use_prefactor = use_prefactor

        # 'load' the other agents actions summation tensor for speed
        self.Omega = self._OtherAgentsActionsSummationTensor()
        self.has_last_statdist = False
        self._last_statedist = jnp.ones(Z) / Z
        
        # use optimized einsum method
        self.opti = opteinsum  

    # =========================================================================
    #   Policy averaging
    # =========================================================================
    @partial(jit, static_argnums=0)    
    def Tss(self, X):
        """Compute average transition model Tss given policy X"""
        # i = 0  # agent i (not needed)
        s = 1  # state s
        sprim = 2  # next state s'
        b2d = list(range(3, 3+self.N))  # all actions

        X4einsum = list(it.chain(*zip(X, [[s, b2d[a]] for a in range(self.N)])))
        args = X4einsum + [self.T, [s]+b2d+[sprim], [s, sprim]]
        return jnp.einsum(*args, optimize=self.opti)

    @partial(jit, static_argnums=0)    
    def Tisas(self, X):
        """Compute average transition model Tisas, given joint policy X"""      
        i = 0  # agent i
        a = 1  # its action a
        s = 2  # the current state
        s_ = 3  # the next state
        j2k = list(range(4, 4+self.N-1))  # other agents
        b2d = list(range(4+self.N-1, 4+self.N-1 + self.N))  # all actions
        e2f = list(range(3+2*self.N, 3+2*self.N + self.N-1))  # all other acts

        sumsis = [[j2k[l], s, e2f[l]] for l in range(self.N-1)]  # sum inds
        otherX = list(it.chain(*zip((self.N-1)*[X], sumsis)))

        args = [self.Omega, [i]+j2k+[a]+b2d+e2f] + otherX\
            + [self.T, [s]+b2d+[s_], [i, s, a, s_]]
        return jnp.einsum(*args, optimize=self.opti)
    
    @partial(jit, static_argnums=0)    
    def Ris(self, X, Risa=None):
        """Compute average reward Ris, given joint policy X""" 
        if Risa is None:  # for speed up
            # Variables      
            i = 0; s = 1; sprim = 2; b2d = list(range(3, 3+self.N))
        
            X4einsum = list(it.chain(*zip(X,
                                    [[s, b2d[a]] for a in range(self.N)])))

            args = X4einsum + [self.T, [s]+b2d+[sprim],
                               self.R, [i, s]+b2d+[sprim], [i, s]]
            return jnp.einsum(*args, optimize=self.opti)
        
        else:  # Compute Ris from Risa 
            i=0; s=1; a=2
            args = [X, [i, s, a], Risa, [i, s, a], [i, s]]
            return jnp.einsum(*args, optimize=self.opti)
       
    @partial(jit, static_argnums=0)    
    def Risa(self, X):
        """Compute average reward Risa, given joint policy X"""
        i = 0; a = 1; s = 2; s_ = 3  # Variables
        j2k = list(range(4, 4+self.N-1))  # other agents
        b2d = list(range(4+self.N-1, 4+self.N-1 + self.N))  # all actions
        e2f = list(range(3+2*self.N, 3+2*self.N + self.N-1))  # all other acts
 
        sumsis = [[j2k[l], s, e2f[l]] for l in range(self.N-1)]  # sum inds
        otherX = list(it.chain(*zip((self.N-1)*[X], sumsis)))

        args = [self.Omega, [i]+j2k+[a]+b2d+e2f] + otherX\
            + [self.T, [s]+b2d+[s_], self.R, [i, s]+b2d+[s_],
               [i, s, a]]
        return jnp.einsum(*args, optimize=self.opti)       
       
    @partial(jit, static_argnums=0)            
    def Vis(self, X, Ris=None, Tss=None, Risa=None):
        """Compute average state values Vis, given joint policy X"""
        # For speed up
        Ris = self.Ris(X, Risa=Risa) if Ris is None else Ris
        Tss = self.Tss(X) if Tss is None else Tss
        
        i = 0  # agent i
        s = 1  # state s
        sp = 2  # next state s'

        n = np.newaxis
        Miss = np.eye(self.Z)[n,:,:] - self.gamma[:, n, n] * Tss[n,:,:]
        
        invMiss = jnp.linalg.inv(Miss)
               
        return self.pre[:,n] * jnp.einsum(invMiss, [i, s, sp], Ris, [i, sp],
                                          [i, s], optimize=self.opti)

    @partial(jit, static_argnums=0)            
    def Qisa(self, X, Risa=None, Vis=None, Tisas=None):
        """Compute average state-action values Qisa, given joint policy X"""
        # For speed up
        Risa = self.Risa(X) if Risa is None else Risa
        Vis = self.Vis(X, Risa=Risa) if Vis is None else Vis
        Tisas = self.Tisas(X) if Tisas is None else Tisas

        nextQisa = jnp.einsum(Tisas, [0,1,2,3], Vis, [0,3], [0,1,2],
                              optimize=self.opti)

        n = np.newaxis
        return self.pre[:,n,n] * Risa + self.gamma[:,n,n]*nextQisa
    
    # =========================================================================
    #   HELPERS
    # =========================================================================
    def statedist(self, X):
        if self.has_last_statdist:
            statedist =  self._jstatedist(X, self._last_statedist)
        else:
            statedist = jnp.array(self._statedist(X))
            self.has_last_statdist = True
            
        self._last_statedist = statedist
        return statedist
        
    @partial(jit, static_argnums=0)  
    def _jstatedist(self, X, pS0, rndkey=42):
        """Compute stationary distribution, given joint policy X"""
        Tss = self.Tss(X)
        pS = compute_stationarydistribution(Tss)
        nrS = jnp.where(pS.mean(0)!=-10, 1, 0).sum()

        @jit
        def single_dist(pS):
            return jnp.max(jnp.where(pS.mean(0)!=-10, jnp.arange(pS.shape[0]),
                                     -1))
        @jit
        def multi_dist(pS):
            # pS0 = self._last_statedist
            ix = jnp.argmin(jnp.linalg.norm(pS.T - pS0, axis=-1))
            return ix
                    
            # @jit
            # def close_stat(pS):
            #     pS0 = jnp.array(self._last_statedist)
            #     ix = np.argmin(jnp.linalg.norm(pS.T - pS0, axis=-1))
            #     return ix
            
            # @jit
            # def rand_stat(pS):
            #     key = jax.random.PRNGKey(rndkey)
            #     choices = jnp.where(pS.mean(0)!=-10, np.arange(pS.shape[0]), -1)
            #     props = jnp.where(pS.mean(0)!=-10, 1, 0)
            #     ix = jax.random.choice(key, choices, (1,), p=props)[0]
            #     return ix
            
            # return jax.lax.cond(self.has_last_statdist, close_stat,
            #                     rand_stat, pS)
            
        ix = jax.lax.cond(nrS == 1, single_dist, multi_dist, pS)

        p = pS[:, ix]
        # self._last_statedist = p
        # self.has_last_statdist = True
        return p
    
    def _statedist(self, X):
        """Compute stationary distribution, given joint policy X"""
        Tss = self.Tss(X)

        # pS = compute_stationarydistribution_alt2(Tss)
        pS = np.array(compute_stationarydistribution(Tss))
        # pS = compute_stationarydistribution(Tss)
        
        pS = pS[:, pS.mean(0)!=-10]
        if len(pS[0]) == 0:  # this happens when the tollerance can distinquish 
            assert False, 'No _statdist return - must not happen'
        elif len(pS[0]) > 1:  # Should not happen, in an ideal world
                # sidenote: This means an ideal world is ergodic ;)
                print("More than 1 state-eigenvector found")
                
                if hasattr(self, '_last_statedist'):  # if last exists
                    # take one that is closesd to last
                    pS0 = self._last_statedist
                    choice = np.argmin(np.linalg.norm(pS.T - pS0, axis=-1))
                    print('taking closest to last')
                    pS = pS[:, choice]
                else:
                    print(pS.round(2))
                    nr = len(pS[0])
                    choice = np.random.randint(nr)
                    print("taking random one: ", choice)
                    pS = pS[:, choice]
                    
        # self._last_statedist = pS.flatten()
        # self.has_last_statdist = True
        return pS.flatten()
       
    def _OtherAgentsActionsSummationTensor(self):
        """
        To sum over the other agents and their respective actions.
        
        For use in Einstein Summation Convention.      
        """
        dim = np.concatenate(([self.N],  # agent i
                              [self.N for _ in range(self.N-1)],  # other agnt
                              [self.M],  # agent a of agent i
                              [self.M for _ in range(self.N)],  # all acts
                              [self.M for _ in range(self.N-1)]))  # other a's
        Omega = np.zeros(dim.astype(int), int)

        for index, _ in np.ndenumerate(Omega):
            I = index[0]
            notI = index[1:self.N]
            A = index[self.N]
            allA = index[self.N+1:2*self.N+1]
            notA = index[2*self.N+1:]

            if len(np.unique(np.concatenate(([I], notI)))) is self.N:
                # all agents indicides are different

                if A == allA[I]:
                    # action of agent i equals some other action
                    cd = allA[:I] + allA[I+1:]  # other actionss
                    areequal = [cd[k] == notA[k] for k in range(self.N-1)]
                    if np.all(areequal):
                        Omega[index] = 1

        return jnp.array(Omega)
    
    def compute_trajectory(self, Xinit, Tmax=100, tolerance=None,
                           verbose=False, **kwargs):
        """
        Compute a trajectory of the evolving learning agents.
           
        Parameters
        ----------
        Xinit : the inital point (either in value or policy space)
        Tmax : the maximum number of iteration steps (default: 100)
        tolerance : to determine if a fix point is reached (default: None) 
        verbose (bool) : whether or not to print info about the run
        """
        traj = []
        t = 0
        X = Xinit.copy()
        fixpreached = False

        while not fixpreached and t < Tmax:
            print(f"\r [computing trajectory] step {t}", end='') if verbose else None 
            traj.append(X)

            X_, TDe = self.TDstep(X)
            if np.any(np.isnan(X_)):
                fixpreached = True
                break

            if tolerance is not None:
                fixpreached = np.linalg.norm(X_ - X) < tolerance
            
            X = X_
            t += 1
                
        print(f" [trajectory computed]") if verbose else None
    
        return np.array(traj), fixpreached

class memomeanfield_partobs_base(memomeanfield_learning_base):
    """
    Base class for
    deterministic policy-average / memory mean field independent (multi-agent) 
    temporal-difference reinforcement learning with partial observability.

    To be used as a base for both, value and policy dynamics.
    """
    
    def __init__(self,
                 TransitionTensor,
                 RewardTensor,
                 ObservationTensor, 
                 DiscountFactors,
                 use_prefactor=False,
                 opteinsum=True,
                 **kwargs):
        """
        Parameters
        ----------
        TransitionTensor : transition model of the environment
        RewardTensor : reward model of the environment
        DiscountFactors : the agents' discount factors
        use_prefactor : use the 1-DiscountFactor prefactor (default: False)
        opteinsum : keyword argument to optimize einsum methods (default: True)
        """
        R = jnp.array(RewardTensor)
        T = jnp.array(TransitionTensor)
        O = jnp.array(ObservationTensor)
    
        # number of agents
        N = R.shape[0]  
        assert len(T.shape[1:-1]) == N, "Inconsistent number of agents"
        assert len(R.shape[2:-1]) == N, "Inconsistent number of agents"
        assert O.shape[0] == N, "Inconsistent number of agents"

        # number of actions for each agent        
        M = T.shape[1] 
        assert np.allclose(T.shape[1:-1], M), 'Inconsisten number of actions'
        assert np.allclose(R.shape[2:-1], M), 'Inconsisten number of actions'
        
        # number of states
        Z = T.shape[0] 
        assert T.shape[-1] == Z, 'Inconsisten number of states'
        assert R.shape[-1] == Z, 'Inconsisten number of states'
        assert R.shape[1] == Z, 'Inconsisten number of states'
        assert O.shape[1] == Z, 'Inconsistent number of states'

        # number of observations
        Q = O.shape[-1]
        
        self.R, self.T, self.O = R, T, O
        self.N, self.M, self.Z, self.Q = N, M, Z, Q
        
        # discount factors
        self.gamma = make_variable_vector(DiscountFactors, N)

        # use (1-DiscountFactor) prefactor to have values on scale of rewards
        self.pre = 1 - self.gamma if use_prefactor else np.ones(N)        
        self.use_prefactor = use_prefactor

        # 'load' the other agents actions summation tensor for speed
        self.Omega = self._OtherAgentsActionsSummationTensor()
        
        # state and obs distribution helpers
        self.has_last_statdist = False
        self._last_statedist = jnp.ones(Z) / Z
        self.has_last_obsdist = False
        self._last_obsdist = jnp.ones((N, Q)) / Q
        
        # use optimized einsum method
        self.opti = opteinsum  

   
    # =========================================================================
    #   Policy averaging
    # =========================================================================
    @partial(jit, static_argnums=0)    
    def Xisa(self, X):
        """
        Compute state-action policy given the current observation-action policy
        """
        i = 0; a = 1; s = 2; o = 4  # variables
        args = [self.O, [i, s, o], X, [i, o, a], [i, s, a]]
        Xisa = jnp.einsum(*args, optimize=self.opti)
    
        # assert np.allclose(Xisa.sum(-1), 1.0), 'Not a policy. Must not happen!'
        return Xisa

    @partial(jit, static_argnums=0)
    def Tss(self, X):
        """Compute average transition model Tss given policy X"""
        Xisa = self.Xisa(X)
        return super().Tss(Xisa)
    
    def Bios(self, X):
        """
        Compute 'belief' that environment is in stats s given agent i
        observes observation o (Bayes Rule)
        """
        pS = self.statedist(X)
        return self._bios(X, pS)
    
    @partial(jit, static_argnums=0)
    def _bios(self, X, pS):
        i, s, o = 0, 1, 2 # variables 

        b = jnp.einsum(self.O, [i,s,o], pS, [s], [i,s,o], optimize=self.opti)
        bsum = b.sum(axis=1, keepdims=True)
        bsum = bsum + (bsum == 0)  # to avoid dividing by zero
        Biso = b / bsum
        Bios = jnp.swapaxes(Biso, 1,-1)
        
        return Bios
        
    @partial(jit, static_argnums=0)
    def fast_Bios(self, X):
        """
        Compute 'belief' that environment is in stats s given agent i
        observes observation o (Bayes Rule)
        
        Unsafe when stationary state distribution is not unique
        (i.e., when policies are too extreme)
        """
        i, s, o = 0, 1, 2 # variables 
        # pS = self.statedist(X) # from full obs base (requires Tss from above)
        pS = self._jstatedist(X, self._last_statedist)

        b = jnp.einsum(self.O, [i,s,o], pS, [s], [i,s,o], optimize=self.opti)
        bsum = b.sum(axis=1, keepdims=True)
        bsum = bsum + (bsum == 0)  # to avoid dividing by zero
        Biso = b / bsum
        Bios = jnp.swapaxes(Biso, 1,-1)
        
        return Bios
    
    @partial(jit, static_argnums=0)    
    def Tioo(self, X, Bios=None, Xisa=None):
        """Compute average transition model Tioo, given joint policy X"""
        # For speed up
        Bios = self.fast_Bios(X) if Bios is None else Bios
        Xisa = self.Xisa(X) if Xisa is None else Xisa
        
        # variables 
        # agent i, state s, next state s_, observation o, next obs o', all acts
        i = 0; s = 1; s_ = 2; o = 3; o_ = 4; b2d = list(range(5, 5+self.N)) 

        Y4einsum = list(it.chain(*zip(Xisa,
                                      [[s, b2d[a]] for a in range(self.N)])))
        
        args = [Bios, [i, o, s]] + Y4einsum + [self.T, [s]+b2d+[s_],
                self.O, [i, s_, o_], [i, o, o_]]
        return jnp.einsum(*args, optimize=self.opti)
    
    @partial(jit, static_argnums=0)    
    def Tioao(self, X, Bios=None, Xisa=None):
        """Compute average transition model Tioao, given joint policy X"""
        # For speed up
        Bios = self.fast_Bios(X) if Bios is None else Bios
        Xisa = self.Xisa(X) if Xisa is None else Xisa
        
        # Variables
        # agent i, act a, state s, next state s_, observation o, next obs o_
        i = 0; a = 1; s = 2; s_ = 3; o = 4; o_ = 5;
        j2k = list(range(6, 6+self.N-1))  # other agents
        b2d = list(range(6+self.N-1, 6+self.N-1 + self.N))  # all actions
        e2f = list(range(5+2*self.N, 5+2*self.N + self.N-1))  # all other acts

        sumsis = [[j2k[l], s, e2f[l]] for l in range(self.N-1)]  # sum inds
        otherY = list(it.chain(*zip((self.N-1)*[Xisa], sumsis)))

        args = [self.Omega, [i]+j2k+[a]+b2d+e2f,
                Bios, [i, o, s]] + otherY + [self.T, [s]+b2d+[s_],
                self.O, [i, s_, o_], [i, o, a, o_]]
        return jnp.einsum(*args, optimize=self.opti)    
    
    @partial(jit, static_argnums=0)    
    def Rioa(self, X, Bios=None, Xisa=None):
        """Compute average reward Riosa, given joint policy X """
        # For speed up
        Bios = self.fast_Bios(X) if Bios is None else Bios
        Xisa = self.Xisa(X) if Xisa is None else Xisa
        
        # Variables
        # agent i, act a, state s, next state s_, observation o
        i = 0; a = 1; s = 2; s_ = 3; o = 4
        j2k = list(range(5, 5+self.N-1))  # other agents
        b2d = list(range(5+self.N-1, 5+self.N-1 + self.N))  # all actions
        e2f = list(range(4+2*self.N, 4+2*self.N + self.N-1))  # all other acts
 
        sumsis = [[j2k[l], s, e2f[l]] for l in range(self.N-1)]  # sum inds
        otherY = list(it.chain(*zip((self.N-1)*[Xisa], sumsis)))

        args = [self.Omega, [i]+j2k+[a]+b2d+e2f, Bios, [i, o, s]] +\
                otherY + [self.T, [s]+b2d+[s_], self.R, [i, s]+b2d+[s_],
                [i, o, a]]

        return jnp.einsum(*args, optimize=self.opti)
    
    @partial(jit, static_argnums=0)        
    def Rio(self, X, Bios=None, Xisa=None, Rioa=None):
        """Compute average reward Rio, given joint policy X"""       
        # For speed up
        if Rioa is None:  # Compute Rio from scratch
            Bios = self.fast_Bios(X) if Bios is None else Bios
            Xisa = self.Xisa(X) if Xisa is None else Xisa
            
            # Variables
            # agent i, state s, next state s_, observation o,  # all actions
            i = 0; s = 1; s_ = 2; o = 3; b2d = list(range(4, 4+self.N)) 
            
            Y4einsum = list(it.chain(*zip(Xisa,
                                    [[s, b2d[a]] for a in range(self.N)])))
            
            args = [Bios, [i, o, s]] + Y4einsum + [self.T, [s]+b2d+[s_],
                    self.R, [i, s]+b2d+[s_], [i, o]]
            return jnp.einsum(*args, optimize=self.opti)
        
        else:  # Compute Rio based on Rioa (should be faster by factor 20)
            i=0; o=1; a=2  # Variables
            args = [X, [i, o, a], Rioa, [i, o, a], [i, o]]
            return jnp.einsum(*args, optimize=self.opti)

    @partial(jit, static_argnums=0)        
    def Vio(self, X,
            Rio=None, Tioo=None, Bios=None, Xisa=None, Rioa=None,
            gamma=None):
        """Compute average observation values Vio, given joint policy X"""
        gamma = self.gamma if gamma is None else gamma 

        # For speed up
        Bios = self.fast_Bios(X) if Bios is None else Bios
        Xisa = self.Xisa(X) if Xisa is None else Xisa
        Rio = self.Rio(X, Bios=Bios, Xisa=Xisa, Rioa=Rioa) if Rio is None\
            else Rio
        Tioo = self.Tioo(X, Bios=Bios, Xisa=Xisa) if Tioo is None\
            else Tioo
        
        i = 0; o = 1; op = 2  # Variables
        n = np.newaxis
        Mioo = np.eye(self.Q)[n,:,:] - gamma[:, n, n] * Tioo
        invMioo = jnp.linalg.inv(Mioo)

        return self.pre[:,n] * jnp.einsum(invMioo, [i, o, op], Rio, [i, op],
                                          [i, o], optimize=self.opti)    

    @partial(jit, static_argnums=0)            
    def Qioa(self, X, Rioa=None, Vio=None, Tioao=None, Bios=None, Xisa=None,
             gamma=None):
        gamma = self.gamma if gamma is None else gamma 
        # For speed up
        Rioa = self.Rioa(X, Bios=Bios, Xisa=Xisa) if Rioa is None\
            else Rioa
        Vio = self.Vio(X, Bios=Bios, Xisa=Xisa, Rioa=Rioa) if Vio is None\
            else Vio        
        Tioao = self.Tioao(X, Bios=Bios, Xisa=Xisa) if Tioao is None\
            else Tioao

        nextQioa = jnp.einsum(Tioao, [0,1,2,3], Vio, [0,3], [0,1,2],
                             optimize=self.opti)
        n = np.newaxis
        return self.pre[:,n,n] * Rioa + gamma[:,n,n]*nextQioa    
    

    # =========================================================================
    #   HELPERS
    # =========================================================================
    def obsdist(self, X):
        if self.has_last_obsdist:
            obsdist =  self._jobsdist(X, self._last_obsdist)
        else:
            obsdist = jnp.array(self._obsdist(X))
            self.has_last_obsdist = True
            
        self._last_obsdist = obsdist
        return obsdist

    @partial(jit, static_argnums=0)  
    def _jobsdist(self, X, pO0, rndkey=42):
        """Compute stationary distribution, given joint policy X"""
        Tioo = self.Tioo(X)
        Dio = jnp.zeros((self.N, self.Q))
        
        for i in range(self.N):
        
            pO = compute_stationarydistribution(Tioo[i])
            nrS = jnp.where(pO.mean(0)!=-10, 1, 0).sum()

            @jit
            def single_dist(pO):
                return jnp.max(jnp.where(pO.mean(0)!=-10,
                                         jnp.arange(pO.shape[0]), -1))
            @jit
            def multi_dist(pO):
                ix = jnp.argmin(jnp.linalg.norm(pO.T - pO0[i], axis=-1))
                return ix
            
            ix = jax.lax.cond(nrS == 1, single_dist, multi_dist, pO)

            Dio = Dio.at[i, :].set(pO[:, ix])

        return Dio
    
    def _obsdist(self, X):
        """Compute stationary distribution, given joint policy X"""
        Tioo = self.Tioo(X)
        Dio = np.zeros((self.N, self.Q))
        
        for i in range(self.N):
            pO = np.array(compute_stationarydistribution(Tioo[i]))
        
            pO = pO[:, pO.mean(0)!=-10]
            if len(pO[0]) == 0:  # this happens when the tollerance can distin.
                assert False, 'No _statdist return - must not happen'
            elif len(pO[0]) > 1:  # Should not happen, in an ideal world
                # sidenote: This means an ideal world is ergodic ;)
                print("More than 1 state-eigenvector found")
                print(pO.round(2))
                nr = len(pO[0])
                choice = np.random.randint(nr)
                print("taking random one: ", choice)
                pO = pO[:, choice]
                        
            Dio[i] = pO.flatten()

        return Dio
    # ===================
    # ======================================================
    #   Additional state based averages
    # =========================================================================
    @partial(jit, static_argnums=0)  
    def Tisas(self, X):
        """Compute average transition model Tisas, given joint policy X"""      
        Xisa = self.Xisa(X)
        return super().Tisas(Xisa)

    @partial(jit, static_argnums=0)  
    def Risa(self, X):
        """Compute average reward Risa, given joint policy X"""
        Xisa = self.Xisa(X)
        return super().Risa(Xisa)

    @partial(jit, static_argnums=0)  
    def Ris(self, X, Risa=None):
        """Compute average reward Ris, given joint policy X""" 
        Xisa = self.Xisa(X)
        return super().Ris(Xisa, Risa=Risa)
    
    @partial(jit, static_argnums=0)  
    def Vis(self, X, Ris=None, Tss=None, Risa=None):
        """Compute average state values Vis, given joint policy X"""
        Xisa = self.Xisa(X)
        Ris = self.Ris(X) if Ris is None else Ris
        Tss = self.Tss(X) if Tss is None else Tss
        return super().Vis(Xisa, Ris=Ris, Tss=Tss, Risa=Risa)

    @partial(jit, static_argnums=0)  
    def Qisa(self, X, Risa=None, Vis=None, Tisas=None):
        """Compute average state-action values Qisa, given joint policy X"""
        Xisa = self.Xisa(X)
        Risa = self.Risa(X) if Risa is None else Risa
        Vis = self.Vis(X) if Vis is None else Vis
        Tisas = self.Tisas(X) if Tisas is None else Tisas
        return super().Qisa(Xisa, Risa=Risa, Vis=Vis, Tisas=Tisas)

# =============================================================================
#   POLICY SPACE 
# ============================================================================= 
class memomeanfield_policybase(memomeanfield_learning_base):
    """
    Base Class for
    deterministic policy-average independent (multi-agent) fully observable
    temporal-difference reinforcement learning in policy space.
    """
    
    def __init__(self, env, learning_rates, discount_factors,
                 choice_intensities=1, **kwargs):
        """
        Parameters
        ----------
        env : environment object
        learning_rates : the learning rate(s) for the agents
        discount_factors : the discount factor(s) for the agents
        choice_intensities : inverse temperature of softmax / exploitation level
        
        Optional Parameters
        --------------------
        use_prefactor (bool) : include the 1-discount_factor prefactor (False)
        opteinsam (bool) : set the optimze keyword in einsums (True)
        """
        self.env = env
        Tt = env.T; assert np.allclose(Tt.sum(-1), 1)
        Rt = env.R    
        super().__init__(Tt, Rt, discount_factors, **kwargs)
        self.F = jnp.array(env.F)

        # learning rates
        self.alpha = make_variable_vector(learning_rates, self.N)

        # intensity of choice
        self.beta = make_variable_vector(choice_intensities, self.N)
        
    @partial(jit, static_argnums=0)
    def TDstep(self, X):
        """
        Temporal-difference learning step in policy space,
        given joint policy X.
        """
        TDe = self.TDerror(X)
        n = jnp.newaxis
        XexpaTDe = X * jnp.exp(self.alpha[:,n,n] * TDe)
        return XexpaTDe / XexpaTDe.sum(-1, keepdims=True), TDe

    def zero_intelligence_policy(self):
        """Policy with equal probabilities."""
        return jnp.ones((self.N, self.Z, self.M)) / float(self.M)

    def random_softmax_policy(self):
        """Softmax policy with random probabilities."""
        expQ = np.exp(np.random.randn(self.N, self.Z, self.M))
        X = expQ / expQ.sum(axis=-1, keepdims=True)
        return jnp.array(X)
            
    def id(self):
        envid = self.env.id() + "__"
        agentsid = f"j{self.__class__.__name__}_"

        if hasattr(self, 'O') and hasattr(self, 'Q'):
            agentsid += 'PartObs_'        
        
        agentsid += f"{str(self.alpha)}_{str(self.gamma)}_{str(self.beta)}"\
            + f"pre{self.use_prefactor}"
        
        return envid + agentsid

class memomeanfield_partobs_policybase(memomeanfield_partobs_base,
                                       memomeanfield_policybase):
    """
    Base Class for
    deterministic policy-average independent (multi-agent) partially observable
    temporal-difference reinforcement learning in policy space.
    """
    
    def __init__(self, env, learning_rates, discount_factors,
                 choice_intensities=1, **kwargs):
        """
        Parameters
        ----------
        env : environment object
        learning_rates : the learning rate(s) for the agents
        discount_factors : the discount factor(s) for the agents
        choice_intensities : inverse temperature of softmax / exploitation level
        """
        self.env = env
        Tt = env.T; assert np.allclose(Tt.sum(-1), 1)
        Rt = env.R
        Ot = env.O    
        super().__init__(Tt, Rt, Ot, discount_factors, **kwargs)
        assert np.allclose(env.F, 0), 'PO learning w final state not def.'

        # learning rates
        self.alpha = make_variable_vector(learning_rates, self.N)
        
        # intensity of choice
        self.beta = make_variable_vector(choice_intensities, self.N)

    def zero_intelligence_policy(self):
        """Policy with equal probabilities."""
        return jnp.ones((self.N, self.Q, self.M)) / float(self.M)

    def random_softmax_policy(self):
        """Softmax policy with random probabilities."""
        expQ = jnp.exp(np.random.randn(self.N, self.Q, self.M))
        return expQ / expQ.sum(axis=-1, keepdims=True)

#   policy space agents 
# -----------------------------------------------------------------------------      
class memomeanfield_partobs_policyAC(memomeanfield_partobs_policybase):
    """
    Class for
    deterministic policy-average independent (multi-agent) partially observable
    temporal-difference actor-critic reinforcement learning in policy space.
    """
    
    @partial(jit, static_argnums=(0,2))
    def TDerror(self, X, norm=False):
        """
        TD error for partially observable policy AC dynamics,
        given joint policy X
        """
        Bios = self.fast_Bios(X)  # for speed up
        Xisa = self.Xisa(X)  # for speed up
        
        R = self.Rioa(X, Bios=Bios, Xisa=Xisa)
        Vio = self.Vio(X, Bios=Bios, Xisa=Xisa, Rioa=R)
        NextV = self.NextVioa(X, Bios=Bios, Xisa=Xisa, Vio=Vio)

        n = jnp.newaxis
        TDe = self.pre[:,n,n]*R + self.gamma[:,n,n]*NextV - Vio[:,:,n]
        TDe *= self.beta[:,n,n]

        TDe = TDe - TDe.mean(axis=2, keepdims=True) if norm else TDe
        return TDe
    
    @partial(jit, static_argnums=0)
    def NextVioa(self, X, Xisa=None, Bios=None, Vio=None, 
                 Tioo=None, Rio=None, Rioa=None):       
        """
        Policy-average next value for agent i, current obs o and act a.
        """
        Xisa = self.Xisa(X) if Xisa is None else Xisa
        Bios = self.fast_Bios(X) if Bios is None else Bios
        Vio = self.Vio(X, Rio=Rio, Tioo=Tioo, Bios=Bios, Xisa=Xisa, Rioa=Rioa)\
            if Vio is None else Vio
        
        i = 0; a = 1; s = 2; s_ = 3; o = 4; o_ = 5  # next observatio 
        j2k = list(range(6, 6+self.N-1))  # other agents
        b2d = list(range(6+self.N-1, 6+self.N-1 + self.N))  # all actions
        e2f = list(range(5+2*self.N, 5+2*self.N + self.N-1))  # all other acts

        sumsis = [[j2k[l], s, e2f[l]] for l in range(self.N-1)]  # sum inds
        otherY = list(it.chain(*zip((self.N-1)*[Xisa], sumsis)))
            
        args = [self.Omega, [i]+j2k+[a]+b2d+e2f, Bios, [i, o, s]] + otherY +\
               [self.T, [s]+b2d+[s_], self.O, [i, s_, o_], 
                Vio, [i, o_], [i, o, a]]
        return jnp.einsum(*args, optimize=self.opti)


# =============================================================================
#   VALUE SPACE 
# =============================================================================    
class multiagentpolicy():
    """General joint policy in tabular form"""

    def __init__(self, epsilon_greedys=None, soft_maxs=None, N=None):
        """
        Policy class to create a general joint policy.
        
        General means either epsilon greedy or soft max with possibly
        heterogeneous parameters for all agents.

        epsilon_greedys : iterable or float
            if iterable: contains exploration parameter for each agent or
                `None` if the agent plays a different policy
            if float: contains exploration parameter assuming homogeneous agents 
        soft_maxs : iterable of float
            if iterable: contains intensity of choice parameter for each egent
                or `None` if the agent plays a different policy  
            if float: contains choice intensity parameter for homogeneous agents 
        N : int
            number of agents, only required if other parameter is givien as a
            single float. Otherwise ignored
        """
        egiter = hasattr(epsilon_greedys, '__iter__')
        smiter = hasattr(soft_maxs, '__iter__')
        if egiter and smiter:
            assert len(epsilon_greedys) == len(soft_maxs),\
                'Both inputs need to be of same length'
            assert N is None, "'N' must not be specified with iterables"
            self.N = len(epsilon_greedys) # Number of agents
        
        elif egiter:  # eps greedy iter, sm not
            self.N = len(epsilon_greedys) # Number of agents
            assert soft_maxs is None, 'Confusing parameter input'
            assert N is None, "'N' must not be specified with iterables"
            soft_maxs = [None] * self.N
        
        elif smiter: # softmax iter, eg not
            self.N = len(soft_maxs) # Number of agents
            assert epsilon_greedys is None, 'Confusing parameter input'
            assert N is None, "'N' must not be specified with iterables"
            epsilon_greedys = [None] * self.N
        
        else: # non has iter
            self.N = N  # Number of agents
            if epsilon_greedys is not None:
                assert soft_maxs is None, 'Confusing parameter input'
                assert type(epsilon_greedys) is float,\
                    'Confusing parameter input'
                epsilon_greedys = [epsilon_greedys] * self.N
                soft_maxs = [None] * self.N
            elif soft_maxs is not None:
                assert epsilon_greedys is None, 'Confusing parameter input'
                assert type(soft_maxs) is float, 'Confusing parameter input'
                soft_maxs = [soft_maxs] * self.N
                epsilon_greedys = [None] * self.N
            else:
                assert False, 'Confusing parameter input'

        # each agent must play a policy
        nones = np.where(np.array([epsilon_greedys, soft_maxs]) == None, 1, 0)
        assert np.all(np.sum(nones, axis=0) == np.ones(self.N)),\
                'Each agent must have a policy'

        # policy indices
        self.epsilongreedy_ix = np.where(np.array(epsilon_greedys) != None)[0]
        self.softmax_ix = np.where(np.array(soft_maxs) != None)[0]

        # exploration values
        self.epsilongreedy_explorations =\
            jnp.array(epsilon_greedys)[self.epsilongreedy_ix].astype(float)
        self.softmax_exploitations =\
            jnp.array(soft_maxs)[self.softmax_ix].astype(float)
    
    @partial(jit, static_argnums=0)
    def action_probabilities(self, Qioa):
        """Transform joint Q values into action probibilities"""
        Xioa = jnp.zeros_like(Qioa)

        epix = self.epsilongreedy_ix
        Xioa = Xioa.at[epix].set(self.epsilongreedy_policy(Qioa[tuple(epix),:,:]))

        smix = self.softmax_ix
        Xioa = Xioa.at[smix].set(self.softmax_policy(Qioa[tuple(smix),:,:]))
        
        return Xioa 
    
    @partial(jit, static_argnums=0)
    def softmax_policy(self, Q):
        """Transform Q values into softmax policy"""
        n = jnp.newaxis 
        beta = self.softmax_exploitations
        expbetaQ = jnp.exp(beta[:,n,n]*Q)
        X = expbetaQ / expbetaQ.sum(-1, keepdims=True)
        
        return X 
    
    @partial(jit, static_argnums=0)
    def epsilongreedy_policy(self, Q):
        """Transform Q values into epsilongreedy policy"""
        n = jnp.newaxis
        X = jnp.zeros_like(Q)
        
        # where are the actions with maximal value?
        maxX = Q == jnp.max(Q, axis=-1, keepdims=True)
        
        # assign 1-eps probability to max actions
        eps = self.epsilongreedy_explorations
        X += (1-eps[:,n,n]) * maxX / maxX.sum(axis=-1, keepdims=True)
        
        # assign eps probability to other actions
        allX = jnp.ones_like(maxX)
        X += eps[:,n,n] * allX / allX.sum(axis=-1, keepdims=True)
        return X 
    

    def id(self):
        id = f"policy_"
        iep = ism = 0
        for i in range(self.N):
            if i in self.epsilongreedy_ix:
                eg = np.array(self.epsilongreedy_explorations[iep])
                id += 'eg'+jnp.array_str(eg, precision=5)+'_'
                iep += 1
            elif i in self.softmax_ix:
                sm = np.array(self.softmax_exploitations[ism])
                id += 'sm'+jnp.array_str(sm, precision=5)+'_'
                ism += 1
            else:
                assert False, 'Must not happen'

        return id[:-1]

class multiagent_epsilongreedy_policy():
    """A multiagent epsilon-greedy policy in tabular form"""
        
    def __init__(self, epsilon_greedys=None, N=None):
        """
        Policy class to create a multiagent epsilon-greedy policy.
        
        epsilon_greedys : iterable or float
            if iterable: contains exploration parameter for each agent or
            if float: contains exploration parameter for all agents 
        N : int
            number of agents, only allowed if `epsilon_greedys` is single float
        """
        egiter = hasattr(epsilon_greedys, '__iter__')
    
        if egiter:  # eps greedy iter, sm not
            self.N = len(epsilon_greedys) # Number of agents
            assert N is None, "'N' must not be specified when iterable is given"
            
        else: 
            self.N = N  # Number of agents
            assert epsilon_greedys is not None, "epsilon value must be given"
            assert type(epsilon_greedys) is float, 'Confusing parameter input'
            epsilon_greedys = [epsilon_greedys] * self.N
            
        # exploration values
        self.epsilongreedy_explorations =\
            jnp.array(epsilon_greedys).astype(float)

 
    @partial(jit, static_argnums=0)
    def action_probabilities(self, Qioa):
        """Transform Q values into epsilongreedy policy"""
        n = jnp.newaxis
        Xioa = jnp.zeros_like(Qioa)
            
        # where are the actions with maximal value?
        maxX = Qioa == jnp.max(Qioa, axis=-1, keepdims=True)
            
        # assign 1-eps probability to max actions
        eps = self.epsilongreedy_explorations
        Xioa += (1-eps[:,n,n]) * maxX / maxX.sum(axis=-1, keepdims=True)
            
        # assign eps probability to other actions
        allX = jnp.ones_like(maxX)
        Xioa += eps[:,n,n] * allX / allX.sum(axis=-1, keepdims=True)
        return Xioa 
    
    def id(self):
        id = f"MAEGpolicy_"
        for i in range(self.N):
            eg = np.array(self.epsilongreedy_explorations[i])
            id += jnp.array_str(eg, precision=5)+'_'
            
        return id[:-1]        
        
class memomeanfield_valuebase(memomeanfield_learning_base):
    """
    Base Class for
    deterministic policy-average independent (multi-agent) fully observable
    temporal-difference reinforcement learning in value space.
    """
    
    def __init__(self, env, learning_rates, discount_factors, policy, **kwargs):
        """
        Parameters
        ----------
        env : environment object
        learning_rates : the learning rate(s) for the agents
        discount_factos : the discount factor(s) for the agents
        policy : policy object
        
        Optional Parameters
        --------------------
        use_prefactor (bool) : include the 1-discount_factor prefactor (False)
        opteinsam (bool) : set the optimze keyword in einsums (True)
        """
        self.env = env  #
        Tt = env.TransitionTensor(); assert np.allclose(Tt.sum(-1), 1)
        Rt = env.RewardTensor()    
        super().__init__(Tt, Rt, discount_factors, **kwargs)
        self.F = jnp.array(env.FinalStates())
        
        # policy
        assert env.N == policy.N, 'Env and Policy must have same `N`'
        self.policy = policy

        # learning rates
        self.alpha = make_variable_vector(learning_rates, self.N)
         
    @partial(jit, static_argnums=0)
    def TDstep(self, Q):
        """
        Temporal-difference learning step in value space,
        given joint state-action values Q.
        """
        TDe = self.TDerror(Q)
        newQ = Q + self.alpha * TDe
        return newQ, TDe    

    def zero_intelligence_values(self):
        """Zero state-action values"""
        return jnp.zeros((self.N, self.Z, self.M))

    def random_values(self):
        """Normal random state-action values"""
        return jnp.array(np.random.randn(self.N, self.Z, self.M))
    
    def id(self):
        envid = self.env.id() + "__"
        agentsid = f"j{self.__class__.__name__}_"\
            + f"{str(self.alpha)}_{str(self.gamma)}_pre{self.use_prefactor}__"
        if hasattr(self, 'O') and hasattr(self, 'Q'):
            agentsid += 'PartObs_' 
        policyid = self.policy.id()
        
        return envid + agentsid + policyid

#   value space agents 
# -----------------------------------------------------------------------------  
class memomeanfield_valueSARSA(memomeanfield_valuebase):
    """
    Class for
    deterministic policy-average independent (multi-agent) fully observable
    temporal-difference SARSA reinforcement learning in value space.
    """

    # We need to construct only the TDerror - rest is already implemented
    
    @partial(jit, static_argnums=0)
    def TDerror(self, Q):
        """
        TD error for fully observable value Q dynamics,
        given joint state-action values Q
        """
        R = self.valRisa(Q)
        NextQ = self.valNextQisa(Q)

        n = np.newaxis
        TDe = self.pre[:,n,n]*R + (1-self.F[n,:,n])*self.gamma[:,n,n]*NextQ - Q
    
        return TDe
        
    @partial(jit, static_argnums=0)
    def valRisa(self, Q):
        """ Average reward Risa, given joint state-action values Q """
        X = self.policy.action_probabilities(Q)
        Risa = self.Risa(X)
        return Risa

    @partial(jit, static_argnums=0)
    def valNextQisa(self, Q):   
        """
        Average max-next state-action values MaxQisa,
        given joint state-action values Q.
        """
        X = self.policy.action_probabilities(Q)
        Qisa = self.Qisa(X)

        i = 0  # agent i
        a = 1  # its action a
        s = 2  # the current state
        sprim = 3  # the next state
        j2k = list(range(4, 4+self.N-1))  # other agents
        b2d = list(range(4+self.N-1, 4+self.N-1 + self.N))  # all actions
        e2f = list(range(3+2*self.N, 3+2*self.N + self.N-1))  # all other acts

        sumsis = [[j2k[l], s, e2f[l]] for l in range(self.N-1)]  # sum inds
        otherX = list(it.chain(*zip((self.N-1)*[X], sumsis)))
            
        NextQisa = jnp.einsum(Qisa, [i, s, a], X, [i, s, a], [i, s])
                    
        args = [self.Omega, [i]+j2k+[a]+b2d+e2f] + otherX +\
            [self.T, [s]+b2d+[sprim], NextQisa, [i, sprim], [i, s, a]]
        return jnp.einsum(*args, optimize=self.opti)          
        
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
#   General helpers
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
def make_variable_vector(variable, length):
    if hasattr(variable, '__iter__'):
        assert len(variable) == length, 'Wrong number given'
        return jnp.array(variable)
    else:
        return jnp.repeat(variable, length)

@jit
def compute_stationarydistribution(Tss):
    """Compute stationary distribution for transition matrix T"""
    # eigenvectors
    oeival, oeivec = jnp.linalg.eig(Tss.T)
    oeival = oeival.real
    oeivec = oeivec.real
    
    get_mask = lambda tol: jnp.abs(oeival - 1) < tol
  
    tolerances = jax.lax.map(lambda x: 0.1**x, jnp.arange(1,16,1))
    masks = jax.lax.map(get_mask, tolerances)
    ix = jnp.max(jnp.where(masks.sum(-1)>=1, jnp.arange(len(masks)), -1))
    mask = masks[ix]
    tol = tolerances[ix]
    
    # obtain stationary distribution
    meivec = jnp.where(mask, oeivec, -1)
    
    dist = meivec / meivec.sum(axis=0, keepdims=True)
    dist = jnp.where(dist < tol, 0, dist)
    dist = dist / dist.sum(axis=0, keepdims=True)
    
    return jnp.where(meivec==-1, -10, dist)


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
#   Code Archive
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# @jit  # similar performance 
# def compute_stationarydistribution_alt1(Tss):
#     """Compute stationary distribution for transition matrix T"""
#     tol = jnp.array([10e-8])
    
#     # eigenvectors
#     oeival, oeivec = jnp.linalg.eig(Tss.T)
#     oeival = oeival.real
#     oeivec = oeivec.real
    
#     get_mask = lambda tol: jnp.abs(oeival - 1) < tol
#     mask = get_mask(tol)


#     # to jit jax requires special control functions 
#     # https://jax.readthedocs.io/en/latest/jax.lax.html#lax-control-flow
#     # here come the functions for the specific cases
#     @jit
#     def refine_mask(mask):
#         tolerances = jax.lax.map(lambda x: 0.1**x, jnp.arange(1,16))
#         masks = jax.lax.map(get_mask, tolerances)
#         ix = jnp.max(jnp.where(masks.sum(-1)==1, jnp.arange(len(masks)), -1))
#         mask = masks[ix]
#         tol = tolerances[ix]
#         return mask #, tol

#     @jit
#     def leave_mask(mask):
#         return mask #, jnp.array([10e-8])

#     # check if there is one EV
#     mask = jax.lax.cond(jnp.sum(mask) != 1, refine_mask, leave_mask, mask)
    

#     # # # obtain stationary distribution
#     # jx = jnp.max(jnp.where(mask==1, jnp.arange(mask.shape[0]), -1))    
#     meivec = jnp.where(mask==1, oeivec, 0).sum(-1)
    
#     dist = meivec / meivec.sum(axis=0, keepdims=True)
#     dist = jnp.where(dist < tol, 0, dist)
#     dist = dist / dist.sum(axis=0, keepdims=True)
    
#     return dist

# # much slower for small systems 
# def compute_stationarydistribution_alt2(
#     Tss, tol=1e-10, adapt_tol=True, verbose=False):
#     """Compute stationary distribution for transition matrix T"""

#     # eigenvectors
#     # oeival, oeivec = np.linalg.eig(Tss.T)
#     oeival, oeivec = la.eig(Tss, right=False, left=True)

#     oeival = oeival.real
#     oeivec = oeivec.real
#     mask = abs(oeival - 1) < tol # which EV is 1?

#     # check if there is one EV
#     printflag = False
#     if adapt_tol and np.sum(mask) != 1:
#         # not ONE eigenvector found AND tolerance adaptation true
#         sign = 2*(np.sum(mask)<1) - 1 # 1 if sum(mask)<1, -1 if sum(mask)>1

#         trial = 0
#         while np.sum(mask)!=1 and tol<1.0 and tol>1e-17 and trial<10:
#             tol = tol * 10**(int(sign))
#             if verbose:
#                 print(f"[detRL-statdist] Adapting tolerance to {tol}")                    
#                 printflag = False

#             mask = abs(oeival - 1) < tol  # reapply mask
#             sign = 2*(np.sum(mask)<1) - 1
#             if np.sum(mask) != 0:
#                 trial += 1

#     # obtain stationary distribution
#     meivec = oeivec[:, mask]
#     dist = meivec / meivec.sum(axis=0, keepdims=True)
#     dist[dist < tol] = 0
#     # dist = dist.at[dist < tol].set(0)
#     dist = dist / dist.sum(axis=0, keepdims=True)

#     if printflag and verbose:
#         print(dist)
#         print(sign)
#     return dist

# # even more slower for small systems 
# def compute_stationarydistribution_alt3(
#     Tss, tol=1e-10, adapt_tol=True, verbose=False):
#     """Compute stationary distribution for transition matrix T"""

#     # eigenvectors
#     oeival, oeivec = jnp.linalg.eig(Tss.T)
    
#     oeival = oeival.real
#     oeivec = oeivec.real
#     mask = abs(oeival - 1) < tol # which EV is 1?

#     # check if there is one EV
#     printflag = False
#     if adapt_tol and jnp.sum(mask) != 1:
#         # not ONE eigenvector found AND tolerance adaptation true
#         sign = 2*(jnp.sum(mask)<1) - 1 # 1 if sum(mask)<1, -1 if sum(mask)>1

#         trial = 0
#         while jnp.sum(mask)!=1 and tol<1.0 and tol>1e-17 and trial<10:
#             tol = tol * 10**(int(sign))
#             if verbose:
#                 print(f"[detRL-statdist] Adapting tolerance to {tol}")                    
#                 printflag = False

#             mask = abs(oeival - 1) < tol  # reapply mask
#             sign = 2*(jnp.sum(mask)<1) - 1
#             if jnp.sum(mask) != 0:
#                 trial += 1

#     # obtain stationary distribution
#     meivec = oeivec[:, mask]
#     dist = meivec / meivec.sum(axis=0, keepdims=True)
#     dist = dist.at[dist < tol].set(0)
#     dist = dist / dist.sum(axis=0, keepdims=True)

#     if printflag and verbose:
#         print(dist)
#         print(sign)
#     return dist