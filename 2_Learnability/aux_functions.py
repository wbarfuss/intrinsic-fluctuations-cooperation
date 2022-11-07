import os
import hashlib
import numpy as np
import matplotlib.pyplot as plt


# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
#   Simulation Scripts
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
def compute_trajectories(AEi, init_values, Tmax=1000):
    qtrajs = []; fprs = []
    lenq = len(init_values)
    for qi, q0 in enumerate(init_values):
        print('\r ', np.round(qi/lenq, 4), end='')
        q = q0.copy()
        
        qtraj, fpr = AEi.compute_trajectory(q, Tmax=Tmax,
                                            tolerance=10e-4, EpsMin=10e-4)
        
        qtrajs.append( qtraj )
        fprs.append( fpr )

    print()
    print('Computed', len(qtrajs), 'trajectories')
    
    return qtrajs, fprs

def check_run(qtrajs, fprs=None):
    if fprs is not None:
        print('Unique fixed points reached:', np.unique(fprs))
    plt.hist([len(traj) for traj in qtrajs], bins=20);
    plt.title('Histrogram of trajectories lengths')



def obtain_simdata(AEi, init_values, verbose=1, Tmax=1000):
    fn = 'data/' + AEi.id()
    fn += '_' + str(_transform_tensor_into_hash(init_values))
    fn += '.npz'
    
    try:
        dat = np.load(fn, allow_pickle=True)
        ddic = dict(zip((k for k in dat), (dat[k] for k in dat)))
        print("Loading ", fn) if verbose else None
    
    except:
        print("Computing ", fn) if verbose else None
        qtrajs, fprs = compute_trajectories(AEi, init_values, Tmax=Tmax)
        check_run(qtrajs, fprs)
        # rtrajs = obtain_rewards(AEi, πtrajs)
        
        ddic = dict(qtrajs=qtrajs, fprs=fprs)
        np.savez_compressed(fn, **ddic)
        dat = np.load(fn, allow_pickle=True)
        ddic = dict(zip((k for k in dat), (dat[k] for k in dat)))
    
    return ddic


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#   helpers for the helpers
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def _transform_tensor_into_hash(tens):
    """Transform tens into a string for filename saving"""
    r = int(hashlib.sha512(str(tens).encode('utf-8')).hexdigest()[:16], 16)
    return r

def _refine_datafolder(datafolder):
    """Check and refine datafolder path"""
    df = os.path.expanduser(datafolder)
    df += '/' if df[-1] != '/' else ''  # make sure path ends as folders do
    return df