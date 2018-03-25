# -*- coding: utf-8 -*-

import numpy as np

from utility import gTrig

def lossSat(cost, m, s,nargout=5):
    D = len(m) # get state dimension
    W = cost.W
    z = cost.z
    SW = s.dot(W);
    
    iSpW = np.linalg.solve((np.eye(D)+SW).T,W.T).T;

#% 1. Expected cost
    # in interval [-1,0]
    L = -np.exp(-(m-z).T.dot(iSpW).dot(m-z)/2)/np.sqrt(np.linalg.det(np.eye(D)+SW))

#% 1a. derivatives of expected cost
    if nargout > 1:
        dLdm = -L*((m-z).T).dot(iSpW)  # wrt input mean
         # wrt input covariance matrix
        dLds = L*(iSpW.dot(m-z).dot((m-z).T) - np.eye(D)).dot(iSpW)/2

#% 2. Variance of cost
    if nargout > 3:
        i2SpW = np.linalg.solve((np.eye(D)+2*SW).T,W.T).T;
        
        r2 = np.exp(-(m-z).T.dot(i2SpW).dot(m-z))/np.sqrt(np.linalg.det(np.eye(D)+2*SW));
        S = r2 - L**2;

        if S < 1e-12:
            S=0 # for numerical reasons


#% 2a. derivatives of variance of cost
    if nargout > 4:
    #  % wrt input mean
        dSdm = -2*r2*((m-z).T).dot(i2SpW)-2*L*dLdm;
    #  % wrt input covariance matrix
        dSds = r2*(2*i2SpW.dot(m-z).dot((m-z).T)-np.eye(D)).dot(i2SpW)-2*L*dLds;

#% 3. inv(s)*cov(x,L)
    if nargout > 6:
        t = W.dot(z) - iSpW.dot(SW.dot(z)+m);
        C = L*t;
        dCdm = t*dLdm - L*iSpW;
        print("error")
#        dCds = -L*(bsxfun(@times,iSpW,permute(t,[3,2,1])) + ...
#                                        bsxfun(@times,permute(iSpW,[1,3,2]),t'))/2;
#        dCds = bsxfun(@times,t,dLds(:)') + reshape(dCds,D,D^2);

    L = 1+L # bring cost to the interval [0,1]
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



def loss_cp(cost, m, s):
    cw = cost.width
    b =  cost.expl
    D0 = np.size(s,1) # state dimension
    D1 = D0 + 2*len(cost.angle) #state dimension (with sin/cos)
    
    
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
        SS = np.kron(np.eye(len(k)),S[np.ix_(i,i)])
        CC = np.kron(C.T,np.eye(len(i)))
        Sdm[ik,:] = SS.dot(dCdm) + CC.dot(Sdm[ii,:])
        Sdm[ki,:] = Sdm[ik,:]
        Sds[ik,:] = SS.dot(dCds) + CC.dot(Sds[ii,:])
        Sds[ki,:] = Sds[ik,:]
        
    L = 0
    dLdm = np.zeros([1,D0])
    dLds = np.zeros([1,D0*D0])
    S2 = 0;
    
    for i in range(len(cw)):                    # scale mixture of immediate costs
        cost.z = target
        cost.W = Q/cw[i]**2;
        [r, rdM, rdS, s2, s2dM, s2dS] = lossSat(cost, M, S, 6);

        L = L + r
        S2 = S2 + s2;
        

        dLdm = dLdm + np.reshape(rdM,[-1,1],order='F').T.dot(Mdm) + np.reshape(rdS,[-1,1],order='F').T.dot(Sdm);
        dLds = dLds + np.reshape(rdM,[-1,1],order='F').T.dot(Mds) + np.reshape(rdS,[-1,1],order='F').T.dot(Sds);


        if (b!=0 or b == []) and abs(s2)>1e-12 :
            L = L + b*np.sqrt(s2);
            print("error")
#            dLdm = dLdm + b/sqrt(s2) * ( s2dM(:)'*Mdm + s2dS(:)'*Sdm )/2;
#            dLds = dLds + b/sqrt(s2) * ( s2dM(:)'*Mds + s2dS(:)'*Sds )/2;

    # normalize
    n = len(cw)
    L = L/n
    dLdm = dLdm/n
    dLds = dLds/n
    S2 = S2/n;

    return L, dLdm, dLds, S2
