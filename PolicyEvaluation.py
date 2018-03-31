# -*- coding: utf-8 -*-

import numpy as np

from utility import gTrig

#飽和コスト
def lossSat(cost, m, s,nargout=5):
    D = len(m) # get state dimension
    iT = cost.iT #W = T^-1
    xtgt = cost.xtgt#目標
    
    #式(47)
    SiT = s.dot(iT);
    #(T^-1)(I+S(T^-1))^-1
    S1 = np.linalg.solve((np.eye(D)+SiT).T,iT.T).T;

    #時刻tの期待コスト ([-1,0]の範囲) 式(46) -> 本関数一番下で、1を足す
    L = -np.exp(-0.5*(m-xtgt).T.dot(S1).dot(m-xtgt))/np.sqrt(np.linalg.det(np.eye(D)+SiT))

    #期待コストの微分
    if nargout > 1:
        #式(48) dE/dμ
        dLdm = -L*((m-xtgt).T).dot(S1)
        #式(49) dE/dΣ
        dLds = 0.5*L*(S1.dot(m-xtgt).dot((m-xtgt).T) - np.eye(D)).dot(S1)

#% 2. Variance of cost
    if nargout > 3:
        i2SpW = np.linalg.solve((np.eye(D)+2*SiT).T,iT.T).T;
        
        r2 = np.exp(-(m-xtgt).T.dot(i2SpW).dot(m-xtgt))/np.sqrt(np.linalg.det(np.eye(D)+2*SiT));
        S = r2 - L**2;

        if S < 1e-12:
            S=0 # for numerical reasons


#% 2a. derivatives of variance of cost
    if nargout > 4:
    #  % wrt input mean
        dSdm = -2*r2*((m-xtgt).T).dot(i2SpW)-2*L*dLdm;
    #  % wrt input covariance matrix
        dSds = r2*(2*i2SpW.dot(m-xtgt).dot((m-xtgt).T)-np.eye(D)).dot(i2SpW)-2*L*dLds;

#% 3. inv(s)*cov(x,L)
    if nargout > 6:
        t = iT.dot(xtgt) - S1.dot(SiT.dot(xtgt)+m);
        C = L*t;
        dCdm = t*dLdm - L*S1;
        print("error")
#        dCds = -L*(bsxfun(@times,S1,permute(t,[3,2,1])) + ...
#                                        bsxfun(@times,permute(S1,[1,3,2]),t'))/2;
#        dCds = bsxfun(@times,t,dLds(:)') + reshape(dCds,D,D^2);

    L = 1+L #式(46)
    if(nargout <= 1):
        return L
    if(nargout <= 3):
        return L,dLdm,dLds
    if(nargout <= 4):
        return L,dLdm,dLds,S
    if(nargout <= 6):
        return L, dLdm, dLds, S, dSdm, dSds
    if(nargout >= 6):
        return L, dLdm, dLds, S, dSdm, dSds, C, dCdm, dCds


#コスト関数
#次状態の分布 = N(x|m,s)
def loss_cp(cost, m, s):
    cw = cost.width
    b =  cost.expl
    D0 = np.size(s,1) # state dimension
    D1 = D0 + 2*np.size(cost.angle) #state dimension (with sin/cos)
    
    
    M = np.zeros([D1,1])
    M[:D0,0:1] = m
    S = np.zeros([D1,D1])
    S[0:D0,0:D0] = s;

    Mdm = np.vstack([np.eye(D0),np.zeros([D1-D0,D0])])
    Sdm = np.zeros([D1*D1,D0])
    Mds = np.zeros([D1,D0*D0]) 
    Sds = np.kron(Mdm,Mdm); # kronecker product
    
    ell = cost.p; # pendulum length
    
    Q = np.zeros([D1,D1])
    Q[np.ix_([0,D0],[0,D0])] = np.array([[1, ell]]).T.dot(np.array([[1, ell]]))
    Q[D0+1,D0+1] = ell**2;
    
    if D1-D0 > 0:
      # augment target
        target = np.vstack([cost.target[:], gTrig(cost.target[:], 0*s, cost.angle)[0]])
        
      # augment state
        i = np.arange(0,D0)
        k = np.arange(D0,D1);
        
        [M[k], S[np.ix_(k,k)], C, mdm, sdm, Cdm, mds, sds, Cds] = gTrig(M[i],S[np.ix_(i,i)],cost.angle,9)

      # compute derivatives (for augmentation)
        X = np.reshape(np.arange(0,D1*D1),[D1, D1],order="F")
        XT = X.T;              # vectorized indices
        I=0*X
        I[np.ix_(i,i)]=1
        ii=X.T[I==1];
        
        I=0*X
        I[np.ix_(k,k)]=1
        kk=X.T[I==1]

        I=0*X
        I[np.ix_(i,k)]=1
        ik=X.T[I.T==1]
        ki=XT.T[I.T==1]

        Mdm[k,:]  = mdm.dot(Mdm[i,:]) + mds.dot(Sdm[ii,:]) # chainrule
        Mds[k,:]  = mdm.dot(Mds[i,:]) + mds.dot(Sds[ii,:])
        Sdm[kk,:] = sdm.dot(Mdm[i,:]) + sds.dot(Sdm[ii,:])
        Sds[kk,:] = sdm.dot(Mds[i,:]) + sds.dot(Sds[ii,:])
        dCdm      = Cdm.dot(Mdm[i,:]) + Cds.dot(Sdm[ii,:])
        dCds      = Cdm.dot(Mds[i,:]) + Cds.dot(Sds[ii,:]);

        S[np.ix_(i,k)] = S[np.ix_(i,i)].dot(C)
        S[np.ix_(k,i)] = S[np.ix_(i,k)].T                      # off-diagonal
        SS = np.kron(np.eye(np.size(k)),S[np.ix_(i,i)])
        CC = np.kron(C.T,np.eye(np.size(i)))
        Sdm[ik,:] = SS.dot(dCdm) + CC.dot(Sdm[ii,:])
        Sdm[ki,:] = Sdm[ik,:]
        Sds[ik,:] = SS.dot(dCds) + CC.dot(Sds[ii,:])
        Sds[ki,:] = Sds[ik,:]
        
    L = 0
    dLdm = np.zeros([1,D0])
    dLds = np.zeros([1,D0*D0])
    S2 = 0;
    
    for i in range(np.size(cw)):                    # scale mixture of immediate costs
        cost.xtgt = target
        cost.iT = Q/(cw[i]**2);
        [r, rdM, rdS, s2, s2dM, s2dS] = lossSat(cost, M, S, 6);

        #累積コストの計算　式(2)
        L = L + r
        S2 = S2 + s2;
        

        dLdm = dLdm + np.reshape(rdM,[-1,1],order='F').T.dot(Mdm) + np.reshape(rdS,[-1,1],order='F').T.dot(Sdm);
        dLds = dLds + np.reshape(rdM,[-1,1],order='F').T.dot(Mds) + np.reshape(rdS,[-1,1],order='F').T.dot(Sds);


        if (abs(s2)>1e-12) :
            L = L + b*np.sqrt(s2);
            dLdm = dLdm + b/np.sqrt(s2) * ( s2dM.reshape([-1,1],order = 'F').T.dot(Mdm) + s2dS.reshape([-1,1],order = 'F').T.dot(Sdm) )/2;
            dLds = dLds + b/np.sqrt(s2) * ( s2dM.reshape([-1,1],order = 'F').T.dot(Mds) + s2dS.reshape([-1,1],order = 'F').T.dot(Sds) )/2;

    # normalize
    n = np.size(cw)
    L = L/n
    dLdm = dLdm/n
    dLds = dLds/n
    S2 = S2/n;

    return L, dLdm, dLds, S2
