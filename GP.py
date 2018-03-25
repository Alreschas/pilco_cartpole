# -*- coding: utf-8 -*-

import numpy as np

def init_gp():
    gp0.oldX = np.empty(0)
    gp2.oldX = np.empty(0)
    gp2d.oldX = np.empty(0)
    gp0d.oldX = np.empty(0)


def maha(nargin,a, b, Q):
    if nargin == 2:
        K = np.sum(a*a,1,keepdims = True) + np.sum(b*b,1,keepdims = True).T -2 * a.dot(b.T);
    else:
        aQ = a.dot(Q)
        K = np.sum(aQ*a,1,keepdims = True) + np.sum(b.dot(Q)*b,1,keepdims = True).T - 2*aQ.dot(b.T);
    return K

def gp0d(nargin,nargout,gpmodel, m, s):

    if nargout < 4:
        print("error")
        [M, S, V] = gp0(3,3,gpmodel, m, s)
        return M,S,V;

    inputs = gpmodel.inputs
    [n, D] = gpmodel.inputs.shape
    E = np.size(gpmodel.targets,1)
    
    #for test
#    gpmodel.hyp = np.array([\
#       [5.479544426113205,   5.531103007660853,   5.465519490848748,   5.600710657391166],\
#       [1.986422822089033,   3.945856914954724,   4.402753819006691,   4.505083086212572],\
#       [3.094041375231762,   2.348837213605796,   2.548113489344849,   2.913900497892119],\
#       [0.685578386885123,   0.543406310836560,   0.163082332006894,   1.050715615218730],\
#       [2.750052705613265,   0.412943068601808,   0.030641788356902,   0.859384483464167],\
#       [4.630573658622718,   3.336565661108991,   2.807800333002539,   3.627360216027113],\
#       [-0.926721103355503,  0.592911270703511,   1.342433007443704,   0.082775701679897,],\
#       [-7.211878280324594, -5.516204260437917,  -4.577579891573320,  -5.370897213576217]\
#       ])
#    
#    s = np.array([\
#     [0.010000000000000,                   0,                   0,                   0,                   0,   0.001393848450685],
#     [                0,   0.010000000000000,                   0,                   0,                   0,   0.001393837669287],
#     [                0,                   0,   0.010000000000000,                   0,                   0,   0.001393775913059],
#     [                0,                   0,                   0,   0.009900663346622,                   0,   0.002774217892993],
#     [                0,                   0,                   0,                   0,   0.000049502904210,   0.000001001324606],
#     [0.001393848450685,   0.001393837669287,   0.001393775913059,   0.002774217892993,   0.000001001324606,   0.002243548948394]
#     ])
    ###
    
    X = gpmodel.hyp;
    
    
#% 1) If necessary, re-compute cached variables
    if np.size(X) != np.size(gp0d.oldX) \
    or np.size(gp0d.iK) == 0 \
    or n != gp0d.oldn \
    or np.sum(np.any(X != gp0d.oldX)):
        gp0d.oldX = np.copy(X)
        gp0d.oldn = n;
        gp0d.K = np.zeros([n,n,E])
        gp0d.iK = np.copy(gp0d.K)
        gp0d.beta = np.zeros([n,E]);
        
  
#  % compute K and inv(K) and beta
        for i in range(E):
            
            inp = inputs/np.exp(X[:D,i]).T;
            #TODO gp2dと微妙に違う
            gp0d.K[:,:,i] = np.exp(2*X[D,i]-maha(2,inp,inp,[])/2);
            if gpmodel.nigp != 0:
                L = np.linalg.cholesky(gp0d.K[:,:,i] + np.exp(2*X[D+1,i])*np.eye(n) + np.diag(gpmodel.nigp[:,i])).T;
            else:
                L = np.linalg.cholesky(gp0d.K[:,:,i] + np.exp(2*X[D+1,i])*np.eye(n));
    
            gp0d.iK[:,:,i] = np.linalg.solve(L.T,np.linalg.solve(L,np.eye(n)));
            gp0d.beta[:,i] = np.linalg.solve(L.T,np.linalg.solve(L,gpmodel.targets[:,i]));

        
#% initializations
    k = np.zeros([n,E]); M = np.zeros([E,1]); V = np.zeros([D,E]); S = np.zeros([E,E]);
    dMds = np.zeros([E,D,D]); dSdm = np.zeros([E,E,D]);
    dSds = np.zeros([E,E,D,D]); dVds = np.zeros([D,E,D,D]); T = np.zeros([D,D]);
 
#% centralize training inputs
    inp = inputs -m.T;
    
 
    
#
#% 2) compute predicted mean and input-output covariance
    for i in range(E):
#  % first some useful intermediate terms
        iL = np.diag(np.exp(-X[:D,i]));
        _in = inp.dot(iL);
        B = iL.dot(s).dot(iL)+np.eye(D)
        LiBL = np.linalg.solve(B.T,iL.T).T.dot(iL)
        t = np.linalg.solve(B.T,_in.T).T
        l = np.exp(-np.sum(_in*t,axis = 1,keepdims = True)/2.)
        lb = l*gp0d.beta[:,i:i+1]
        tL = t.dot(iL)
        tlb = tL * lb
        c = np.exp(2*X[D,i])/np.sqrt(np.linalg.det(B));
        M[i,0] = c * np.sum(lb,axis = 0,keepdims = True)  # predicted mean
        V[:,i:i+1] = tL.T.dot(lb).dot(c)        # input-output covariance
        dMds[i,:,:] = c*tL.T.dot(tlb)/2-LiBL*M[i,0]/2;
        
        #TODO 要確認 gp2dと微妙に違う
        for d in range(D):
            dVds[d,i,:,:] = c*(tL*tL[:,d:d+1]).T.dot(tlb)/2 - LiBL*V[d,i]/2 - (V[:,i:i+1].dot(LiBL[d:d+1,:]) + LiBL[:,d:d+1].dot(V[:,i:i+1].T))/2

#        for d in range(D):
#            dVds[d,i,:,:] = c*(t * t[:,d:d+1]).T.dot(tlb)/2 - iR*V[d,i]/2 - V[:,i:i+1].dot(iR[d:d+1,:])/2 -iR[:,d:d+1].dot(V[:,i:i+1].T)/2;

        k[:,i:i+1] = 2*X[D,i]-np.sum(_in*_in,1,keepdims = True)/2;

    dMdm = V.T# derivatives w.r.t m
    dVdm = 2 * np.transpose(dMds,[1,0,2])
    
    iell2 = np.exp(-2*gpmodel.hyp[:D,:])
    inpiell2 = np.empty([inp.shape[0],inp.shape[1],iell2.shape[1]])
    for p in range(iell2.shape[1]):
        inpiell2[:,:,p] = (inp * np.transpose(np.atleast_3d(iell2),[2,0,1])[:,:,p])

#% 3) predictive covariance matrix (non-central moments)
    for i in range(E):
        ii = inpiell2[:,:,i];
  
        for j in range(i+1): # if i==j: diagonal elements of S; see Marc's thesis around eq. (2.26)
            R = s.dot(np.diag(iell2[:,i]+iell2[:,j]))+np.eye(D)
            t = 1/np.sqrt(np.linalg.det(R));
            ij = inpiell2[:,:,j]
            L = np.exp((k[:,i:i+1]+k[:,j:j+1].T)+maha(3,ii,-ij,np.linalg.solve(R,s)/2)) # called Q in thesis

            if(i == j):
                iKL = gp0d.iK[:,:,i]*L 

                s1iKL = np.sum(iKL,0,keepdims = True)
                s2iKL = np.sum(iKL,1,keepdims = True)
                S[i,j] = t*(gp0d.beta[:,i:i+1].T.dot(L).dot(gp0d.beta[:,i:i+1]) - np.sum(s1iKL))
                zi = np.linalg.solve(R.T,ii.T).T
                                
                bibLi = L.T.dot(gp0d.beta[:,i:i+1])*gp0d.beta[:,i:i+1]
                cbLi = L.T.dot(gp0d.beta[:,i:i+1] * zi)
                r = (bibLi.T.dot(zi)*2 - (s2iKL.T + s1iKL).dot(zi))*t;
        
                for d in range(D):                
                    T[d:d+1,:d+1] = 2*(zi[:,:d+1].T.dot(zi[:,d:d+1]*bibLi) \
                    + cbLi[:,:d+1].T.dot(zi[:,d:d+1] * gp0d.beta[:,i:i+1]) \
                    - zi[:,:d+1].T.dot(zi[:,d:d+1]*s2iKL) \
                    - zi[:,:d+1].T.dot(iKL.dot(zi[:,d:d+1])))[:,0]
                    
                    T[:d+1,d:d+1] = T[d:d+1,:d+1].T;

            else:
                zi = np.linalg.solve(R.T,ii.T).T
                zj = np.linalg.solve(R.T,ij.T).T;
                S[i,j] = gp0d.beta[:,i:i+1].T.dot(L).dot(gp0d.beta[:,j:j+1])*t; 
                S[j,i] = S[i,j];
      
                bibLj = L.dot(gp0d.beta[:,j:j+1])*gp0d.beta[:,i:i+1]; 
                bjbLi = L.T.dot(gp0d.beta[:,i:i+1])*gp0d.beta[:,j:j+1];
                cbLi = L.T.dot(gp0d.beta[:,i:i+1] * zi);
                cbLj = L.dot(gp0d.beta[:,j:j+1] * zj);
      
                r = (bibLj.T.dot(zi)+bjbLi.T.dot(zj))*t;
                
                for d in range(D):
                    T[d:d+1,:d+1] = (zi[:,:d+1].T.dot(zi[:,d:d+1]*bibLj) 
                    + cbLi[:,:d+1].T.dot(zj[:,d:d+1]*gp0d.beta[:,j:j+1])\
                    + zj[:,:d+1].T.dot(zj[:,d:d+1]*bjbLi) \
                    + cbLj[:,:d+1].T.dot(zi[:,d:d+1]*gp0d.beta[:,i:i+1]))[:,0];
                    
                    T[:d+1,d:d+1] = T[d:d+1,:d+1].T; 
                    
      
            
            dSdm[i,j,:] = r - M[i,0]*(dMdm[j:j+1,:]) - M[j,0]*(dMdm[i:i+1,:]);
            dSdm[j,i,:] = dSdm[i,j,:];

            T = (t*T-S[i,j]*np.linalg.solve(R.T,np.diag((np.exp(-2*X[:D,i:i+1])+np.exp(-2*X[:D,j:j+1]))[:,0]).T))/2;
            T = T - np.reshape(M[i,0]*dMds[j,:,:] + M[j,0]*dMds[i,:,:],[D,D],order = 'F');
            

            dSds[i,j,:,:] = T
            dSds[j,i,:,:] = T
    
        #loop end j
    
        S[i,i] = S[i,i] + np.exp(2*X[D,i]); 
        
    #loop end i

#% 4) centralize moments
    S = S - M.dot(M.T);
    
#%S(diag(S)<0,diag(S)<0) = 1e-6;

#% 5) Vectorize derivatives
    dMds=np.reshape(dMds,[E, D*D],order = 'F');
 
    dSds=np.reshape(dSds,[E*E, D*D],order = 'F');
    dSdm=np.reshape(dSdm,[E*E, D],order = 'F');
    
    dVds=np.reshape(dVds,[D*E, D*D],order = 'F');
    dVdm=np.reshape(dVdm,[D*E, D],order = 'F');
    
    return M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds  


def gp0(nargin,nargout,gpmodel,m,s):
    
    inputs = gpmodel.inputs
    [n, D] = gpmodel.inputs.shape
    E = np.size(gpmodel.targets,1)
    
    X = gpmodel.hyp;
    
    if np.size(X) != np.size(gp0.oldX) \
    or np.size(gp0.iK) == 0 \
    or n != gp0.oldn \
    or np.sum(np.any(X != gp0.oldX)):
        gp0.oldX = np.copy(X)
        gp0.oldn = n;
        gp0.K = np.zeros([n,n,E])
        gp0.iK = np.copy(gp0.K)
        gp0.beta = np.zeros([n,E]);

        for i in range(E):
            
            inp = inputs/np.exp(X[:D,i]).T;
            #TODO gp0と微妙に違う
            gp0.K[:,:,i] = np.exp(2*X[D,i]-maha(2,inp,inp,[])/2);
            if gpmodel.nigp != 0:
                L = np.linalg.cholesky(gp0.K[:,:,i] + np.exp(2*X[D+1,i])*np.eye(n) + np.diag(gpmodel.nigp[:,i])).T;
            else:
                L = np.linalg.cholesky(gp0.K[:,:,i] + np.exp(2*X[D+1,i])*np.eye(n));
    
            gp0.iK[:,:,i] = np.linalg.solve(L.T,np.linalg.solve(L,np.eye(n)));
            gp0.beta[:,i] = np.linalg.solve(L.T,np.linalg.solve(L,gpmodel.targets[:,i]));

    k = np.zeros([n,E]); M = np.zeros([E,1]); V = np.zeros([D,E]); S = np.zeros([E,E]);
    inp = inputs -m.T;

    for i in range(E):
#  % first some useful intermediate terms
        iL = np.diag(np.exp(-X[:D,i]));
        _in = inp.dot(iL);
        B = iL.dot(s).dot(iL)+np.eye(D)

        t = np.linalg.solve(B.T,_in.T).T
        l = np.exp(-np.sum(_in*t,axis = 1,keepdims = True)/2.)
        lb = l*gp0.beta[:,i:i+1]
        tiL = t.dot(iL)
        c = np.exp(2*X[D,i])/np.sqrt(np.linalg.det(B));

        M[i,0] = c * np.sum(lb,axis = 0,keepdims = True)  # predicted mean
        V[:,i:i+1] = tiL.T.dot(lb).dot(c)        # input-output covariance
        k[:,i:i+1] = 2*X[D,i]-np.sum(_in*_in,1,keepdims = True)/2;

    for i in range(E):
        ii = (inp/np.exp(2*X[:D,i].T));
  
        for j in range(i+1): # if i==j: diagonal elements of S; see Marc's thesis around eq. (2.26)
            R = s.dot(np.diag(np.exp(-2*X[:D,i])+np.exp(-2*X[:D,j])))+np.eye(D)
            t = 1/np.sqrt(np.linalg.det(R));
            ij = (inp/np.exp(2*X[:D,j:j+1].T));
            L = np.exp((k[:,i:i+1]+k[:,j:j+1].T)+maha(3,ii,-ij,np.linalg.solve(R,s)/2)) # called Q in thesis
            if i==j:
                S[i,i] = t * (gp0.beta[:,i:i+1].T.dot(L).dot(gp0.beta[:,j:j+1]) - np.sum(gp0.iK[:,:,i]*L))
            else:
                S[i,j] = gp0.beta[:,i:i+1].T.dot(L).dot(gp0.beta[:,j:j+1])*t; 
                S[j,i] = S[i,j];
            
        S[i,i] = S[i,i] + np.exp(2*X[D,i]); 
        
    #loop end i

#% 4) centralize moments
    S = S - M.dot(M.T);
    return M,S,V        


def gp1(nargin,nargout,gpmodel, m, s):
    if np.size(gpmodel.induce) == 0:
        [M, S, V] = gp0(3,3,gpmodel, m, s); 
        return M,S,V
    print('error')
    return 0,0,0

def gp1d(nargin,nargout,gpmodel, m, s):
    if nargout < 4:
        [M, S, V] = gp1(3,3,gpmodel, m, s)
        return M,S,V;
    
    if np.size(gpmodel.induce) == 0:
        [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds] = gp0d(3,9,gpmodel, m, s); 
        return M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds
    
    print('error')
    return 0,0,0,0,0,0,0,0,0


def gp2(gpmodel, m, s):

    inputs = gpmodel.inputs
    targets = gpmodel.targets
    X = gpmodel.hyp;
    
    D = np.size(inputs,1);       # number of examples and dimension of input space

    [n, E] = targets.shape# number of examples and number of outputs    

    if np.size(X) != np.size(gp2.oldX) \
    or np.size(gp2.iK) == 0 \
    or n != gp2.oldn \
    or np.sum(np.any(X != gp2.oldX))\
    or np.sum(np.any(gp2.oldIn != inputs)) \
    or np.sum(np.any(gp2.oldOut != targets)):
        gp2.oldX = np.copy(X)
        gp2.oldIn = np.copy(inputs)
        gp2.oldOut = np.copy(targets)
        gp2.oldn = n;
        gp2.K = np.zeros([n,n,E])
        gp2.iK = np.copy(gp2.K)
        gp2.beta = np.zeros([n,E]);
        
        for i in range(E):
            inp = inputs/np.exp(X[:D,i]).T;
            gp2.K[:,:,i] = np.exp(2*X[D,i])-maha(2,inp,inp,[])/2;
            if gpmodel.nigp != 0:
                L = np.linalg.cholesky(gp2.K[:,:,i] + np.exp(2*X[D+1,i])*np.eye(n) + np.diag(gpmodel.nigp[:,i])).T;
            else:
                L = np.linalg.cholesky(gp2.K[:,:,i] + np.exp(2*X[D+1,i])*np.eye(n));

            gp2.iK[:,:,i] = np.linalg.solve(L.T,np.linalg.solve(L,np.eye(n)));
            gp2.beta[:,i] = np.linalg.solve(L.T,np.linalg.solve(L,gpmodel.targets[:,i]));
        
    k = np.zeros([n,E]); M = np.zeros([E,1]); V = np.zeros([D,E]); S = np.zeros([E,E]);
    
    inp = inputs -m.T;
    
    for i in range(E):
        iL = np.diag(np.exp(-X[:D,i]));
        _in = inp.dot(iL);
        B = iL.dot(s).dot(iL)+np.eye(D)

        t = np.linalg.solve(B.T,_in.T).T
        l = np.exp(-np.sum(_in*t,axis = 1,keepdims = True)/2.)
        lb = l*gp2.beta[:,i:i+1]
        tL = t.dot(iL)
        c = np.exp(2*X[D,i])/np.sqrt(np.linalg.det(B));
        
        M[i,0] = c * np.sum(lb,axis = 0,keepdims = True)  # predicted mean
        V[:,i:i+1] = tL.T.dot(lb).dot(c)        # input-output covariance
        
        k[:,i:i+1] = 2*X[D,i]-np.sum(_in*_in,1,keepdims = True)/2;
        
    for i in range(E):
        ii = (inp/np.exp(2*X[:D,i].T));
  
        for j in range(i+1): # if i==j: diagonal elements of S; see Marc's thesis around eq. (2.26)
            R = s.dot(np.diag(np.exp(-2*X[:D,i])+np.exp(-2*X[:D,j])))+np.eye(D)
            t = 1/np.sqrt(np.linalg.det(R));
            ij = (inp/np.exp(2*X[:D,j:j+1].T));
            L = np.exp((k[:,i:i+1]+k[:,j:j+1].T)+maha(3,ii,-ij,np.linalg.solve(R,s)/2)) # called Q in thesis
            
            S[i,j] = t * (gp2.beta[:,i:i+1].T).dot(L).dot(gp2.beta[:,j:j+1])
            S[j,i] = S[i,j];
            
        S[i,i] = S[i,i] + 1e-06;
    S = S - M.dot(M.T);
    return M,S,V


def gp2d(nargout,gpmodel, m, s):
    inputs = gpmodel.inputs
    targets = gpmodel.targets
    X = gpmodel.hyp;
    
    if nargout < 4:
        print("error")
        [M, S, V] = gp2(gpmodel, m, s)
        return

    D = np.size(inputs,1);       # number of examples and dimension of input space

    [n, E] = targets.shape# number of examples and number of outputs
    X = np.reshape(X, [D+2, E],order='F');

#% 1) If necessary, re-compute cached variables
    if np.size(X) != np.size(gp2d.oldX) \
    or np.size(gp2d.iK) == 0 \
    or n != gp2d.oldn \
    or np.sum(np.any(X != gp2d.oldX))\
    or np.sum(np.any(gp2d.oldIn != inputs)) \
    or np.sum(np.any(gp2d.oldOut != targets)):
        gp2d.oldX = np.copy(X)
        gp2d.oldIn = np.copy(inputs)
        gp2d.oldOut = np.copy(targets)
        gp2d.oldn = n;
        gp2d.K = np.zeros([n,n,E])
        gp2d.iK = np.copy(gp2d.K)
        gp2d.beta = np.zeros([n,E]);
        
  
#  % compute K and inv(K) and beta
        for i in range(E):
            inp = inputs/np.exp(X[:D,i]).T;
            gp2d.K[:,:,i] = np.exp(2*X[D,i])-maha(2,inp,inp,[])/2;
            if gpmodel.nigp != 0:
                L = np.linalg.cholesky(gp2d.K[:,:,i] + np.exp(2*X[D+1,i])*np.eye(n) + np.diag(gpmodel.nigp[:,i])).T;
            else:
                L = np.linalg.cholesky(gp2d.K[:,:,i] + np.exp(2*X[D+1,i])*np.eye(n));
    
            gp2d.iK[:,:,i] = np.linalg.solve(L.T,np.linalg.solve(L,np.eye(n)));
            gp2d.beta[:,i] = np.linalg.solve(L.T,np.linalg.solve(L,gpmodel.targets[:,i]));
            
#% initializations
    k = np.zeros([n,E]); M = np.zeros([E,1]); V = np.zeros([D,E]); S = np.zeros([E,E]);
    dMds = np.zeros([E,D,D]); dSdm = np.zeros([E,E,D]); r = np.zeros([1,D]);
    dSds = np.zeros([E,E,D,D]); dVds = np.zeros([D,E,D,D]); T = np.zeros([D,D]);
    tlbdi = np.zeros([n,D]); dMdi = np.zeros([E,n,D]); dMdt = np.zeros([E,n,E]);
    dVdt = np.zeros([D,E,n,E]); dVdi = np.zeros([D,E,n,D]); dSdt = np.zeros([E,E,n,E]);
    dSdi = np.zeros([E,E,n,D]); dMdX = np.zeros([E,D+2,E]); dSdX = np.zeros([E,E,D+2,E]);
    dVdX = np.zeros([D,E,D+2,E]); Z = np.zeros([n,D]);
    bdX = np.zeros([n,E,D]); kdX = np.zeros([n,E,D+1]);

#% centralize training inputs
    inp = inputs -m.T;
#
#% 2) compute predicted mean and input-output covariance
    for i in range(E):
#  % first some useful intermediate terms
        K2 = gp2d.K[:,:,i]+np.exp(2*X[D+1,i])*np.eye(n);# K + sigma^2*I
        inp2 = inputs/np.exp(X[:D,i].T);
        ii = inputs/np.exp(2*X[:D,i].T)
        R = s+np.diag(np.exp(2*X[:D,i]));
        L = np.diag(np.exp(-X[:D,i]));
        B = L.dot(s).dot(L)+np.eye(D)
        iR = np.linalg.solve(B.T,L.T).T.dot(L)
        t = inp.dot(iR);
        l = np.atleast_2d(np.exp(-np.sum(t*inp,1)/2)).T
        lb = l*gp2d.beta[:,i:i+1]
        tliK = t.T.dot(l*gp2d.iK[:,:,i])
        liK = np.linalg.solve(K2,l)
        tlb = t * lb
        
        c = np.exp(2*X[D,i])/np.sqrt(np.linalg.det(R))*np.exp(np.sum(X[:D,i]));
        
        detdX = np.diag(np.linalg.det(R)*iR.T * 2*np.exp(2*X[:D,i]))    # d(det R)/dX
        cdX = -0.5*(c/np.linalg.det(R))*detdX.T+ c*np.ones([1,D]);      # derivs w.r.t length-scales

        dldX = l*(t*2*np.exp(2*X[:D,i].T))*t/2;
  
        M[i,0] = np.sum(lb)*c;                                           # predicted mean
  
        iK2beta = np.atleast_2d(np.linalg.solve(K2,gp2d.beta[:,i])).T;
        dMds[i,:,:] = c*t.T.dot(tlb)/2-iR*M[i,0]/2;

        dMdX[i,D+1,i] = -c*np.sum(l*(2*np.exp(2*X[D+1,i])*(iK2beta)))
        dMdX[i,D,i] = -dMdX[i, i*(D+2)+D+1];
        
  
        dVdX[:,i,D+1,i] = -((l*(2*np.exp(2*X[D+1,i])*iK2beta)).T.dot(t)*c)
        dVdX[:,i,D,i] = -dVdX[:,i,D+1,i];

  
        dsi = -inp2*2*inp2    # d(sum(inp2.*inp2,2))/dX
        dslb = np.zeros([1,D]);
  
        for d in range(D):
            sqdi = gp2d.K[:,:,i]*(ii[:,d:d+1]-ii[:,d:d+1].T)
            sqdiBi = sqdi.dot(gp2d.beta[:,i:i+1]);
            tlbdi[:,d:d+1] = (sqdi.dot(liK)*gp2d.beta[:,i:i+1] + sqdiBi*liK);
            tlbdi2 = -tliK.dot((-sqdi * gp2d.beta[:,i:i+1]).T-np.diag(sqdiBi[:,0]));
            dVdi[:,i,:,d] = c*(iR[:,d:d+1]*lb.T - (t * tlb[:,d:d+1]).T + tlbdi2)

            dsqdX = (dsi[:,d:d+1] + dsi[:,d:d+1].T) + 4.*inp2[:,d:d+1]*inp2[:,d:d+1].T
            dKdX = -gp2d.K[:,:,i]*dsqdX/2;                               #dK/dX(1:D)
            dKdXbeta = dKdX.dot(gp2d.beta[:,i:i+1]);
            bdX[:,i,d:d+1] = -np.linalg.solve(K2,dKdXbeta);                        # dbeta/dX
            dslb[0,d] = -liK.T.dot(dKdXbeta) + gp2d.beta[:,i:i+1].T.dot(dldX[:,d:d+1]);
            dlb = dldX[:,d:d+1]*gp2d.beta[:,i:i+1] + l*bdX[:,i,d:d+1];
            dtdX = inp.dot(-(iR[:,d:d+1] * 2*np.exp(2*X[d,i])*iR[d:d+1,:]));
            dlbt = lb.T.dot(dtdX) + dlb.T.dot(t);
            dVdX[:,i,d,i:i+1] = (dlbt.T*c + cdX[0,d]*(lb.T.dot(t)).T);

        dMdi[i,:,:] = c*(tlbdi - tlb)
        dMdt[i,:,i] = c*liK.T;
        dMdX[i,:D,i] = cdX*np.sum(gp2d.beta[:,i:i+1]*l) + c*dslb;
        v = (inp/np.exp(X[:D,i:i+1].T));
        k[:,i] = 2*X[D,i]-np.sum(v*v,1)/2;
        V[:,i:i+1] = t.T.dot(lb).dot(c)        # input-output covariance
        

        for d in range(D):
            dVds[d,i,:,:] = c*(t * t[:,d:d+1]).T.dot(tlb)/2 - iR*V[d,i]/2 - V[:,i:i+1].dot(iR[d:d+1,:])/2 -iR[:,d:d+1].dot(V[:,i:i+1].T)/2;
            kdX[:,i,d:d+1] = (v[:,d:d+1] * v[:,d:d+1]);
            
  
        dVdt[:,i,:,i] = c*tliK;
        kdX[:,i,D] = 2*np.ones([1,n]);  # pre-computation for later
        

    dMdm = V.T# derivatives w.r.t m
    dVdm = 2 * np.transpose(dMds,[1,0,2])
    

#% 3) predictive covariance matrix (non-central moments)
    for i in range(E):
        K2 = gp2d.K[:,:,i]+np.exp(2*X[D+1,i])*np.eye(n);
        ii = (inp/np.exp(2*X[:D,i].T));
  
        for j in range(i+1): # if i==j: diagonal elements of S; see Marc's thesis around eq. (2.26)
            R = s.dot(np.diag(np.exp(-2*X[:D,i])+np.exp(-2*X[:D,j])))+np.eye(D)
            t = 1/np.sqrt(np.linalg.det(R));

            if 1/np.linalg.cond(R) < 1e-15:
                print('R-matrix in gp2d ill-conditioned')
            
            iR = np.linalg.solve(R,np.eye(D))
            ij = (inp/np.exp(2*X[:D,j:j+1].T));
            L = np.exp((k[:,i:i+1]+k[:,j:j+1].T)+maha(3,ii,-ij,np.linalg.solve(R,s)/2)) # called Q in thesis
            A = gp2d.beta[:,i:i+1].dot(gp2d.beta[:,j:j+1].T)
            A = A*L
            ssA = np.sum(A)
            
            S[i,j] = t*ssA
            S[j,i] = S[i,j];
                        
            zzi = ii.dot(np.linalg.solve(R,s))
            zzj = ij.dot(np.linalg.solve(R,s))
            zi = np.linalg.solve(R.T,ii.T).T
            zj = np.linalg.solve(R.T,ij.T).T
    
            tdX  = -0.5*t*np.sum(iR.T*(s * (-2*np.exp(-2*X[:D,i:i+1].T)-2*np.exp(-2*X[:D,i].T))),axis=0);
            tdXi = -0.5*t*np.sum(iR.T*(s * -2*np.exp(-2*X[:D,i:i+1].T)),axis=0);
            tdXj = -0.5*t*np.sum(iR.T*(s * -2*np.exp(-2*X[:D,j:i+1].T)),axis=0);
            bLiKi = gp2d.iK[:,:,j].dot(L.T.dot(gp2d.beta[:,i:i+1]))
            bLiKj = gp2d.iK[:,:,i].dot(L.dot(gp2d.beta[:,j:j+1]))

    
            Q2 = np.linalg.solve(R,s)/2
            aQ = ii.dot(Q2)
            bQ = ij.dot(Q2)
    
            for d in range(D):      
                Z[:,d:d+1] = np.exp(-2*X[d,i])*(A.dot(zzj[:,d:d+1]) + np.sum(A,1,keepdims=True)*(zzi[:,d:d+1] - inp[:,d:d+1]))\
                 + np.exp(-2*X[d,j])*((zzi[:,d:d+1]).T.dot(A) + np.sum(A,0,keepdims=True)*(zzj[:,d:d+1] - inp[:,d:d+1]).T).T;
                 
    
                Q = (inp[:,d:d+1] - inp[:,d:d+1].T)
                B = gp2d.K[:,:,i]*Q;
                Z[:,d:d+1] = Z[:,d:d+1]+np.exp(-2*X[d,i]) *(B.dot(gp2d.beta[:,i:i+1])*bLiKj+gp2d.beta[:,i:i+1]*(B.dot(bLiKj)));

                if i!=j:
                    B = gp2d.K[:,:,j]*Q
      
                Z[:,d:d+1] = Z[:,d:d+1]+np.exp(-2*X[d,j])*(bLiKi*(B.dot(gp2d.beta[:,j:j+1]))+B.dot(bLiKi)*gp2d.beta[:,j:j+1]);
                B = (zi[:,d:d+1] + zj[:,d:d+1].T)*A;
                r[0,d] = np.sum(B)*t
                T[d:d+1,:d+1] = (np.sum(zi[:,:d+1].T.dot(B),1,keepdims=True) + np.sum(B.dot(zj[:,:d+1]),0,keepdims = True).T).T;
                T[:d+1,d:d+1] = T[d:d+1,:d+1].T;
      
                if i==j:
                    RTi =  (s *(-2*np.exp(-2*X[:D,i:i+1].T)-2*np.exp(-2*X[:D,j:j+1].T)))
                    diRi = -np.linalg.solve(R,(RTi[:,d:d+1] * iR[d:d+1,:]))
                else:
                    RTi = (s * (-2*np.exp(-2*X[:D,i:i+1].T)));
                    RTj = (s * (-2*np.exp(-2*X[:D,j:j+1].T)));
                    diRi = -np.linalg.solve(R,(RTi[:,d:d+1] * iR[d:d+1,:]));
                    diRj = -np.linalg.solve(R,(RTj[:,d:d+1] * iR[d:d+1,:]));
                    QdXj = diRj.dot(s)/2; # dQ2/dXj
                QdXi = diRi.dot(s)/2; # dQ2/dXj

                if i==j:
                    daQi = ii.dot(QdXi) + (-2*ii[:,d:d+1] * Q2[d:d+1,:]) # d(ii*Q)/dXi
                    dsaQi = np.sum(daQi*ii,1,keepdims=True) - 2*aQ[:,d:d+1]*ii[:,d:d+1]
                    dsaQj = np.copy(dsaQi)
                    dsbQi = np.copy(dsaQi)
                    dsbQj = np.copy(dsbQi)
                    dm2i = -2*daQi.dot(ii.T) + 2*((aQ[:,d:d+1] * ii[:,d:d+1].T) + (ii[:,d:d+1] * aQ[:,d:d+1].T));
                    dm2j = np.copy(dm2i); # -2*aQ*ij'/di
                else:
                    dbQi = ij.dot(QdXi);  # d(ij*Q)/dXi
                    dbQj = ij.dot(QdXj) + (-2*ij[:,d:d+1] * Q2[d:d+1,:]); # d(ij*Q)/dXj
                    daQi = ii.dot(QdXi) + (-2*ii[:,d:d+1] * Q2[d:d+1,:]); # d(ii*Q)/dXi
                    daQj = ii.dot(QdXj); # d(ii*Q)/dXj
        
                    dsaQi = np.sum(daQi*ii,1,keepdims=True) - 2*aQ[:,d:d+1]*ii[:,d:d+1];
                    dsaQj = np.sum(daQj*ii,1,keepdims=True);
                    dsbQi = np.sum(dbQi*ij,1,keepdims=True);
                    dsbQj = np.sum(dbQj*ij,1,keepdims=True) - 2*bQ[:,d:d+1]*ij[:,d:d+1];
                    dm2i = -2*daQi.dot(ij.T); # second part of the maha(..) function wrt Xi
                    dm2j = -2*ii.dot(dbQj.T); #second part of the maha(..) function wrt Xj
      
                dm1i = (dsaQi + dsbQi.T) # first part of the maha(..) function wrt Xi
                dm1j = (dsaQj + dsbQj.T) # first part of the maha(..) function wrt Xj
                dmahai = dm1i-dm2i;
                dmahaj = dm1j-dm2j;
      
                if i==j:
                    LdXi = L*(dmahai + (kdX[:,i,d:d+1] + kdX[:,j,d:d+1].T));
                    dSdX[i,i,d,i] = gp2d.beta[:,i:i+1].T.dot(LdXi).dot(gp2d.beta[:,j:j+1]);
                else:
                    LdXi = L*(dmahai + (kdX[:,i,d:d+1] + np.zeros([n,1]).T));
                    LdXj = L*(dmahaj + (np.zeros([n,1]) + kdX[:,j,d:d+1].T));
                    dSdX[i,j,d,i] = gp2d.beta[:,i:i+1].T.dot(LdXi).dot(gp2d.beta[:,j:j+1]);
                    dSdX[i,j,d,j] = gp2d.beta[:,i:i+1].T.dot(LdXj).dot(gp2d.beta[:,j:j+1]);

            if i==j:
                dSdX[i,i,:D,i:i+1] = np.reshape(dSdX[i,i,:D,i],[D,1],order = 'F') \
                + np.reshape(bdX[:,i,:],[n,D],order = 'F').T.dot(L+L.T).dot(gp2d.beta[:,i:i+1]);
                
                dSdX[i,i,:D,i] = np.reshape(t*(dSdX[i,i,:D,i:i+1]),[D,1],order = 'F').T + tdX.dot(ssA)

                dSdX[i,i,D+1,i] = 2*np.exp(2*X[D+1,i])*t* \
                (-np.sum(gp2d.beta[:,i:i+1]*bLiKi,axis = 0,keepdims = True)\
                 -np.sum(gp2d.beta[:,i:i+1]*bLiKi,axis = 0,keepdims = True));

            else:
                dSdX[i,j,:D,i] = np.reshape(dSdX[i,j,:D,i],[D,1],order = 'F') + np.reshape(bdX[:,i,:],[n,D],order = 'F').T*(L.dot(gp2d.beta[:,j:j+1]));
                dSdX[i,j,:D,j] = np.reshape(dSdX[i,j,:D,j],[D,1],order = 'F') + np.reshape(bdX[:,j,:],[n,D],order = 'F').T*(L.T.dot(gp2d.beta[:,i:i+1]));
                dSdX[i,j,:D,i] = np.reshape(t*dSdX[i,j,:D,i],[D,1],order = 'F').T + tdXi.dot(ssA);
                dSdX[i,j,:D,j] = np.reshape(t*dSdX[i,j,:D,j],[D,1],order = 'F').T + tdXj.dot(ssA);
                dSdX[i,j,D+1,i] = 2*np.exp(2*X[D+1,i])*t*(-gp2d.beta[:,i:i+1].T.dot(bLiKj));
                dSdX[i,j,D+1,j] = 2*np.exp(2*X[D+1,j])*t*(-gp2d.beta[:,j:j+1].T.dot(bLiKi));
            
            dSdm[i,j,:] = r - M[i,0]*(dMdm[j:j+1,:]) - M[j,0]*(dMdm[i:i+1,:]);
            dSdm[j,i,:] = dSdm[i,j,:];
            T = (t*T-S[i,j]*np.linalg.solve(R.T,np.diag((np.exp(-2*X[:D,i:i+1])+np.exp(-2*X[:D,j:j+1]))[:,0]).T))/2;

            T = T - np.reshape(M[i,0]*dMds[j,:,:] + M[j,0]*dMds[i,:,:],[D,D],order = 'F');

            dSds[i,j,:,:] = T
            dSds[j,i,:,:] = T
    
            if i==j:
                dSdt[i,i,:,i] = np.linalg.solve(K2.T,(gp2d.beta[:,i:i+1].T.dot(L+L.T)).T).T*t - 2*dMdt[i,:,i]*M[i,0];
                dSdX[i,j,:,i] = np.reshape(dSdX[i,j,:,i],[1,D+2],order='F') - M[i,0]*dMdX[j,:,j]-M[j]*dMdX[i,:,i];
            else:
                dSdt[i,j,:,i] = np.linalg.solve(K2.T,(gp2d.beta[:,j:j+1].T.dot(L.T)).T).T*t - dMdt[i,:,i]*M[j,0];
                dSdt[i,j,:,j] = np.linalg.solve((K[:,:,j]+np.exp(2*X[D+1,j])*np.eye(n)).T,(gp2d.beta[:,i:i+1].T.dot(L)).T).T*t - dMdt[j,:,j]*M[i,0]
                dSdt[j,i,:,:] = dSdt[i,j,:,:]
                dSdX[i,j,:,j] = np.reshape(dSdX[i,j,:,j],[1,D+2],order = 'F') - M[i,0]*dMdX[j,:,j]
                dSdX[i,j,:,i] = np.reshape(dSdX[i,j,:,i],[1,D+2],order = 'F') - M[j,0]*dMdX[i,:,i]
    
            dSdi[i,j,:,:] = Z*t - np.reshape(M[i,0]*dMdi[j,:,:] + dMdi[i,:,:]*M[j,0],[n,D],order = 'F');
            dSdi[j,i,:,:] = dSdi[i,j,:,:]
            dSdX[j,i,:,:] = dSdX[i,j,:,:]
        #loop end j
    
        S[i,i] = S[i,i] + 1e-06;    # add small diagonal jitter for numerical reasons
        
    #loop end i

    dSdX[:,:,D,:] = -dSdX[:,:,D+1,:];
    dSdX[:,:,D,:] = -dSdX[:,:,D+1,:];

#% 4) centralize moments
    S = S - M.dot(M.T);
    
#%S(diag(S)<0,diag(S)<0) = 1e-6;

#% 5) Vectorize derivatives
    dMds=np.reshape(dMds,[E, D*D],order = 'F');
    dSdm=np.reshape(dSdm,[E*E, D],order = 'F');
    dSds=np.reshape(dSds,[E*E, D*D],order = 'F');
    
    dVdm=np.reshape(dVdm,[D*E, D],order = 'F');
    dVds=np.reshape(dVds,[D*E, D*D],order = 'F');
    
    dMdi=np.reshape(dMdi,[E,-1],order = 'F')
    dMdt=np.reshape(dMdt,[E,-1],order = 'F')  
    dMdX=np.reshape(dMdX,[E,-1],order = 'F')
    
    dSdi=np.reshape(dSdi,[E*E,-1],order = 'F');
    dSdt=np.reshape(dSdt,[E*E,-1],order = 'F');
    dSdX=np.reshape(dSdX,[E*E,-1],order = 'F');
    
    dVdi=np.reshape(dVdi,[D*E,-1],order = 'F');
    dVdt=np.reshape(dVdt,[D*E,-1],order = 'F');
    dVdX=np.reshape(dVdX,[D*E,-1],order = 'F');

    return M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds, dMdi, dSdi, dVdi,dMdt, dSdt, dVdt, dMdX, dSdX, dVdX
  