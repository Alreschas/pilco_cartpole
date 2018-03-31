# -*- coding: utf-8 -*-

import numpy as np
import scipy.optimize
import copy

from utility import unwrap, rewrap
from utility import gTrig


def f2(nargout, *varargin):
    if nargout == 0:
        f2.p = varargin
        f2.F = f2.p[0]
    else:
        s = rewrap(f2.p[1], varargin[0])

        [fx, dfx] = f2.F(s, f2.p[2][0], f2.p[2][1], f2.p[2][2], f2.p[2][3], f2.p[2][4], f2.p[2][5], f2.p[2][6])

        return fx, dfx

# own


class PolicyOptimizer:
    def __init__(self, x0, *varargin):
        self.F = value
        self.dfx = np.empty(0)
        f2(0, self.F, x0, varargin)

    def targetFunc(self, x):
        [fx, dfx] = f2(2, x)
        self.dfx = dfx
        return fx

    def targetFunc_dev(self, x):
        if(np.size(self.dfx) != 0):
            dfx = self.dfx
        else:
            [fx, dfx] = f2(2, x)
        return dfx[:, 0]


def sliceModel(dynmodel, n, ii, D1, D2, D3):  # separate sub-dynamics
    if (dynmodel.sub != 0):
        dyn = dynmodel.sub[n]
        do = dyn.dyno
        D = np.size(ii) + D1 - D2
        if (dyn.dyni != []):
            di = dyn.dyni
        else:
            di = []
        if (dyn.dynu != []):
            du = dyn.dynu
        else:
            du = []
        if (dyn.dynj != []):
            dj = dyn.dynj
        else:
            dj = []
        i = np.hstack([ii[di], D1 + du, D2 + dj])
        k = D2 + do
        dyn.inputs = np.hstack([dynmodel.inputs[:, np.hstack([di, D + du])], dynmodel.target[:, dj]]);  # inputs
        dyn.target = dynmodel.target[:, do];  # targets
    else:
        dyn = dynmodel
        k = np.arange(D2, D3)
        i = ii

    return dyn, i, k


def fillIn(nargin, nargout, S, C, mdm, sdm, Cdm, mds, sds, Cds, Mdm, Sdm, Mds, Sds, Mdp, Sdp, dCdp, i, j, k, D):
    if k is ():
        return
    X = np.reshape(np.arange(D * D), [D, D], order='F')
    XT = X.T  # vectorized indices
    I = 0 * X
    I[np.ix_(i, i)] = 1
    ii = XT[(I == 1).T].T
    I = 0 * X
    I[np.ix_(k, k)] = 1
    kk = XT[(I == 1).T].T
    I = 0 * X
    I[np.ix_(j, i)] = 1
    ji = XT[(I == 1).T].T
    I = 0 * X
    I[np.ix_(j, k)] = 1
    jk = XT[(I == 1).T].T
    kj = X[(I == 1).T].T

    Mdm[k, :] = mdm.dot(Mdm[i, :]) + mds.dot(Sdm[ii, :])  # chainrule
    Mds[k, :] = mdm.dot(Mds[i, :]) + mds.dot(Sds[ii, :])
    Sdm[kk, :] = sdm.dot(Mdm[i, :]) + sds.dot(Sdm[ii, :])
    Sds[kk, :] = sdm.dot(Mds[i, :]) + sds.dot(Sds[ii, :])
    dCdm = Cdm.dot(Mdm[i, :]) + Cds.dot(Sdm[ii, :])
    dCds = Cdm.dot(Mds[i, :]) + Cds.dot(Sds[ii, :])

    if nargin < 19 and nargout > 5:
        Mdp[k, :] = mdm.dot(Mdp[i, :]) + mds.dot(Sdp[ii, :]);
        Sdp[kk, :] = sdm.dot(Mdp[i, :]) + sds.dot(Sdp[ii, :]);
        dCdp = Cdm.dot(Mdp[i, :]) + Cds.dot(Sdp[ii, :]);
    elif nargout > 5:
        aa = np.size(k)
        bb = aa**2
        cc = np.size(C)

        mdp = np.zeros([D, np.size(Mdp, 1)])
        sdp = np.zeros([D * D, np.size(Mdp, 1)])
        mdp[k, :] = np.reshape(Mdp, [aa, -1], order='F')
        Mdp = mdp

        sdp[kk, :] = np.reshape(Sdp, [bb, -1], order='F')
        Sdp = sdp

        Cdp = np.reshape(dCdp, [cc, -1], order='F')
        dCdp = Cdp

    q = S[np.ix_(j, i)].dot(C)
    S[np.ix_(j, k)] = q
    S[np.ix_(k, j)] = q.T  # off-diagonal
    SS = np.kron(np.eye(np.size(k)), S[np.ix_(j, i)])
    CC = np.kron(C.T, np.eye(np.size(j)))
    Sdm[jk, :] = SS.dot(dCdm) + CC.dot(Sdm[ji, :]);
    Sdm[kj, :] = Sdm[jk, :];
    Sds[jk, :] = SS.dot(dCds) + CC.dot(Sds[ji, :]);
    Sds[kj, :] = Sds[jk, :];

    if nargout > 5:
        Sdp[jk, :] = SS.dot(dCdp) + CC.dot(Sdp[ji, :])
        Sdp[kj, :] = Sdp[jk, :];

    if(nargout == 5):
        return S, Mdm, Mds, Sdm, Sds
    if(nargout == 7):
        return S, Mdm, Mds, Sdm, Sds, Mdp, Sdp


def propagate(m, s, plant, dynmodel, policy):
    angi = plant.angi
    poli = plant.poli
    dyni = plant.dyni
    difi = plant.difi

    D0 = len(m)  # size of the input mean
    D1 = D0 + 2 * len(angi)  # length after mapping all angles to sin/cos
    D2 = D1 + len(policy.maxU)  # length after computing control signal
    D3 = D2 + D0  # length after predicting
    M = np.zeros([D3, 1])
    M[0:D0] = m
    S = np.zeros([D3, D3])
    S[0:D0, 0:D0] = s  # init M and S

    i = np.arange(D0)
    j = np.arange(D0)
    k = np.arange(D0, D1)

    ss = np.zeros([D1, D1])

    [M[k], S[np.ix_(k, k)], C] = gTrig(M[i], S[np.ix_(i, i)], angi, 3)
    q = S[np.ix_(j, i)].dot(C)
    S[np.ix_(j, k)] = q
    S[np.ix_(k, j)] = q.T

    sn2 = np.exp(2 * dynmodel.hyp[-1, :])
    sn2[difi] = sn2[difi] / 2

    mm = np.zeros([D1, 1])
    mm[i] = M[i]
    ss[np.ix_(i, i)] = S[np.ix_(i, i)] + np.diag(sn2)
    [mm[k], ss[np.ix_(k, k)], C] = gTrig(mm[i], ss[np.ix_(i, i)], angi, 3)  # noisy state measurement
    q = ss[np.ix_(j, i)].dot(C)
    ss[np.ix_(j, k)] = q
    ss[np.ix_(k, j)] = q.T

    i = poli
    j = np.arange(D1)
    k = np.arange(D1, D2)

    [M[k], S[np.ix_(k, k)], C] = policy.fcn[0](3, policy.fcn[1][0], policy.fcn[1][1], policy, mm[i], ss[np.ix_(i, i)])
    q = S[np.ix_(j, i)].dot(C)
    S[np.ix_(j, k)] = q
    S[np.ix_(k, j)] = q.T

    ii = np.hstack([dyni, np.arange(D1, D2)])
    j = np.arange(D2)

    if(dynmodel.sub != 0):
        Nf = np.size(dynmodel.sub)
    else:
        Nf = 1

    for n in range(Nf):                               # potentially multiple dynamics models
        [dyn, i, k] = sliceModel(dynmodel, n, ii, D1, D2, D3)
        j = np.setdiff1d(j, k)

        [M[k], S[np.ix_(k, k)], C] = dyn.fcn(3, 3, dyn, M[i], S[np.ix_(i, i)])

        q = S[np.ix_(j, i)].dot(C)
        S[np.ix_(j, k)] = q
        S[np.ix_(k, j)] = q.T

        j = np.hstack([j, k])

    P = np.hstack([np.zeros([D0, D2]), np.eye(D0)])
    P[np.ix_(difi, difi)] = np.eye(np.size(difi))

    Mnext = P.dot(M)
    Snext = P.dot(S).dot(P.T)
    Snext = (Snext + Snext.T) / 2

    return Mnext, Snext


def propagated(m, s, plant, dynmodel, policy, nargout=8):

    if nargout <= 2:  # just predict, no derivatives
        [Mnext, Snext] = propagate(m, s, plant, dynmodel, policy)
        return Mnext, Snext

    angi = plant.angi
    poli = plant.poli
    dyni = plant.dyni
    difi = plant.difi

    D0 = np.size(m)  # size of the input mean
    D1 = D0 + 2 * np.size(angi)  # length after mapping all angles to sin/cos
    D2 = D1 + np.size(policy.maxU)  # length after computing control signal
    D3 = D2 + D0  # length after predicting
    M = np.zeros([D3, 1])
    M[0:D0] = m
    S = np.zeros([D3, D3])
    S[0:D0, 0:D0] = s  # init M and S

    Mdm = np.vstack([np.eye(D0), np.zeros([D3 - D0, D0])])
    Sdm = np.zeros([D3 * D3, D0])
    Mds = np.zeros([D3, D0 * D0])
    Sds = np.kron(Mdm, Mdm)
    X = np.reshape(np.arange(D3 * D3), [D3, D3], order='F')
    XT = X.T
    Sds = (Sds + Sds[XT.reshape(np.size(XT), order='F'), :]) / 2;
    X = np.reshape(np.arange(D0 * D0), [D0, D0])
    XT = X.T
    Sds = (Sds + Sds[:, XT.reshape(np.size(XT), order='F')]) / 2;

    i = np.arange(D0)
    j = np.arange(D0)
    k = np.arange(D0, D1)

    ss = np.zeros([D1, D1])

    [M[k], S[np.ix_(k, k)], C, mdm, sdm, Cdm, mds, sds, Cds] = gTrig(M[i], S[np.ix_(i, i)], angi, 9)

    [S, Mdm, Mds, Sdm, Sds] = fillIn(16, 5, S, C, mdm, sdm, Cdm, mds, sds, Cds, Mdm, Sdm, Mds, Sds, [], [], [], i, j, k, D3)

    sn2 = np.exp(2 * dynmodel.hyp[-1, :])
    sn2[difi] = sn2[difi] / 2

    mm = np.zeros([D1, 1])
    mm[i] = M[i]
    ss[np.ix_(i, i)] = S[np.ix_(i, i)] + np.diag(sn2)
    [mm[k], ss[np.ix_(k, k)], C] = gTrig(mm[i], ss[np.ix_(i, i)], angi, 3)  # noisy state measurement
    q = ss[np.ix_(j, i)].dot(C)
    ss[np.ix_(j, k)] = q
    ss[np.ix_(k, j)] = q.T

    i = poli
    j = np.arange(D1)
    k = np.arange(D1, D2)

    [M[k], S[np.ix_(k, k)], C, mdm, sdm, Cdm, mds, sds, Cds, Mdp, Sdp, Cdp] = policy.fcn[0](12, policy.fcn[1][0], policy.fcn[1][1], policy, mm[i], ss[np.ix_(i, i)])

    [S, Mdm, Mds, Sdm, Sds, Mdp, Sdp] = fillIn(19, 7, S, C, mdm, sdm, Cdm, mds, sds, Cds, Mdm, Sdm, Mds, Sds, Mdp, Sdp, Cdp, i, j, k, D3)

    ii = np.hstack([dyni, np.arange(D1, D2)])
    j = np.arange(D2)

    if(dynmodel.sub != 0):
        Nf = np.size(dynmodel.sub)
    else:
        Nf = 1

    for n in range(Nf):                               # potentially multiple dynamics models
        [dyn, i, k] = sliceModel(dynmodel, n, ii, D1, D2, D3)
        j = np.setdiff1d(j, k)

        [M[k], S[np.ix_(k, k)], C, mdm, sdm, Cdm, mds, sds, Cds] = dyn.fcn(3, 9, dyn, M[i], S[np.ix_(i, i)])

        [S, Mdm, Mds, Sdm, Sds, Mdp, Sdp] = fillIn(18, 7, S, C, mdm, sdm, Cdm, mds, sds, Cds, Mdm, Sdm, Mds, Sds, Mdp, Sdp, [], i, j, k, D3)

        j = np.hstack([j, k])

    P = np.hstack([np.zeros([D0, D2]), np.eye(D0)])
    P[np.ix_(difi, difi)] = np.eye(np.size(difi))

    Mnext = P.dot(M)
    Snext = P.dot(S).dot(P.T)
    Snext = (Snext + Snext.T) / 2

    PP = np.kron(P, P)
    dMdm = P.dot(Mdm)
    dMds = P.dot(Mds)
    dMdp = P.dot(Mdp)

    dSdm = PP.dot(Sdm)
    dSds = PP.dot(Sds)
    dSdp = PP.dot(Sdp)

    X = np.reshape(np.arange(D0 * D0), [D0, D0], order='F')
    XT = X.T  # symmetrize dS
    dSdm = (dSdm + dSdm[XT.reshape([-1]), :]) / 2;
    dMds = (dMds + dMds[:, XT.reshape([-1])]) / 2;

    dSds = (dSds + dSds[XT.reshape([-1]), :]) / 2;
    dSds = (dSds + dSds[:, XT.reshape([-1])]) / 2;

    dSdp = (dSdp + dSdp[XT.reshape([-1]), :]) / 2;

    return Mnext, Snext, dMdm, dSdm, dMds, dSds, dMdp, dSdp


def value(p, m0, S0, dynmodel, policy, plant, cost, H):

    policy.param = copy.deepcopy(p)
    p = unwrap(p)
    
#    print(p)
    

    dp = 0 * p
    m = m0
    S = S0
    L = np.zeros([1, H])

    dmOdp = np.zeros([np.size(m0, 0), len(p)])
    dSOdp = np.zeros([np.size(m0, 0) * np.size(m0, 0), len(p)])

    for t in range(H):  # for all time steps in horizon
        [m, S, dmdmO, dSdmO, dmdSO, dSdSO, dmdp, dSdp] = plant.prop(m, S, plant, dynmodel, policy)  # get next state

        dmdp = dmdmO.dot(dmOdp) + dmdSO.dot(dSOdp) + dmdp
        dSdp = dSdmO.dot(dmOdp) + dSdSO.dot(dSOdp) + dSdp

        [L[0, t], dLdm, dLdS, tmp] = cost.fcn(cost, m, S)  # predictive cost

        L[0, t] = (cost.gamma**t) * L[0, t]                             # discount
        dp = dp + (cost.gamma**t) * (dLdm.reshape([-1, 1], order='F').T.dot(dmdp) + dLdS.reshape([-1, 1], order='F').T.dot(dSdp)).T

        dmOdp = np.copy(dmdp)
        dSOdp = np.copy(dSdp)  # bookkeeping

    J = np.sum(L, keepdims=True)
    dJdp = dp

    print(J)
    
    return J, dJdp


def calcCost(cost, M, S):

    H = np.size(M, 1)           # horizon length
    L = np.zeros([1, H])
    SL = np.zeros([1, H])


#% for each time step, compute the expected cost and its variance
    for h in range(H):
        [L[0, h], d1, d2, SL[0, h]] = cost.fcn(cost, M[:, h:h + 1], S[:, :, h]);

    sL = np.sqrt(SL)

    return L, sL


def pred(policy, plant, dynmodel, m, s, H):

    D = np.size(m)
    S = np.zeros([D, D, H + 1])
    M = np.zeros([D, H + 1])
    M[:, 0:1] = m
    S[:, :, 0] = s
    for i in range(H):
        [m, s] = plant.prop(m, s, plant, dynmodel, policy, 2)
        M[:, i + 1:i + 2] = m[-D:, 0:1];
        S[:, :, i + 1] = s[-D:, -D:];
    return M, S


def learnPolicy(policy, dynmodel, plant, cost, H, mu0Sim, S0Sim):

    #    unwrap(policy.p)

    X_fmt = policy.param
    X0 = unwrap(X_fmt)[:,0]
    
    po_own = PolicyOptimizer(X_fmt, mu0Sim, S0Sim, dynmodel, policy, plant, cost, H)

    result = scipy.optimize.minimize(po_own.targetFunc, X0, jac=po_own.targetFunc_dev, method='BFGS',options={'maxiter':200,'disp':True})

    policy.param = rewrap(policy.param,result['x'])
    fX3 = result['fun']
    print('optimize succeed:',fX3)

    [M, Sigma] = pred(policy, plant, dynmodel, mu0Sim[:, 0:1], S0Sim, H)

    [meanCost, stdCost] = calcCost(cost, M, Sigma)
    return M, Sigma, meanCost, stdCost

if __name__ == '__main__':
    from GP import init_gp
    init_gp()
    
    class GpModel:
        def __init__(self):
            self.nigp = 0
            self.inputs = np.arange(1,51).reshape([10,5],order='F')
            self.targets = np.arange(1,11).reshape([10,1],order='F')
            self.hyp = np.arange(1,8).reshape([7,1],order='F');
    gpmodel = GpModel()
    m = np.arange(1,6).reshape([5,1],order='F')
    s = np.arange(1,26).reshape([5,5],order='F')