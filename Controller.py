# -*- coding: utf-8 -*-
import numpy as np

from GP import gp2d
from GP import gp2


def conCat(nargout,con, sat, policy, m, s):
    maxU=policy.maxU; # amplitude limit of control signal
    E=np.size(maxU);   # dimension of control signal
    D=np.size(m);      # dimension of input
    
    F=D+E
    j=np.arange(D,F)
    i=np.arange(D)
    # initialize M and S
    M = np.zeros([F,1])
    M[i] = m
    S = np.zeros([F,F])
    S[np.ix_(i,i)] = s
    
    if nargout < 4:
        [M[j], S[np.ix_(j,j)], Q] = con(3,policy, m, s);  # compute unsquashed control signal v
        q = S[np.ix_(i,i)].dot(Q)
        S[np.ix_(i,j)] = q
        S[np.ix_(j,i)] = q.T  # compute joint covariance S=cov(x,v)
        [M, S, R] = sat(4,3,M, S, j, maxU);         # compute squashed control signal u
        C = np.hstack([np.eye(D),Q]).dot(R);                       # inv(s)*cov(x,u)
        if nargout == 1:
            return M
    else:
        Mdm = np.zeros([F,D])
        Sdm = np.zeros([F*F,D])
        Mdm[:D,:D] = np.eye(D);
        
        Mds = np.zeros([F,D*D])
        Sds = np.kron(Mdm,Mdm);
        
        
        X = np.reshape(np.arange(F*F),[F, F],order = 'F');
        XT = X.T                  # vectorized indices
        I=0*X
        I[np.ix_(j,j)]=1
        jj=X[I==1].T;
        I=0*X
        I[np.ix_(i,j)]=1
        ij=X[I==1].T
        ji=XT[I==1].T;
        
#          % 1. Unsquashed controller --------------------------------------------------
#        con(12,policy, m, s)
        [M[j], S[np.ix_(j,j)], Q, Mdm[j,:], Sdm[jj,:], dQdm, Mds[j,:], Sds[jj,:], dQds, Mdp, Sdp, dQdp] = con(12,policy, m, s);
        q = S[np.ix_(i,i)].dot(Q)
        S[np.ix_(i,j)] = q
        S[np.ix_(j,i)] = q.T;  # compute joint covariance S=cov(x,v)
        
#          % update the derivatives
        SS = np.kron(np.eye(E),S[np.ix_(i,i)])
        QQ = np.kron(Q.T,np.eye(D))
        
        Sdm[ij,:] = SS.dot(dQdm)
        Sdm[ji,:] = Sdm[ij,:]
        
        Sds[ij,:] = SS.dot(dQds) + QQ
        Sds[ji,:] = Sds[ij,:]
                
#          % 2. Apply Saturation -------------------------------------------------------
        [M, S, R, MdM, SdM, RdM, MdS, SdS, RdS] = sat(4,9,M, S, j, maxU)
          
#          % apply chain-rule to compute derivatives after concatenation
        dMdm = MdM.dot(Mdm) + MdS.dot(Sdm)
        dMds = MdM.dot(Mds) + MdS.dot(Sds)
        dSdm = SdM.dot(Mdm) + SdS.dot(Sdm)
        dSds = SdM.dot(Mds) + SdS.dot(Sds)
        dRdm = RdM.dot(Mdm) + RdS.dot(Sdm)
        dRds = RdM.dot(Mds) + RdS.dot(Sds)

            
        dMdp = MdM[:,j].dot(Mdp) + MdS[:,jj].dot(Sdp)
        dSdp = SdM[:,j].dot(Mdp) + SdS[:,jj].dot(Sdp)
        dRdp = RdM[:,j].dot(Mdp) + RdS[:,jj].dot(Sdp)
          
        C = np.hstack([np.eye(D), Q]).dot(R); # inv(s)*cov(x,u)
#          % update the derivatives
        RR = np.kron(R[j,:].T,np.eye(D))
        QQ = np.kron(np.eye(E),np.hstack([np.eye(D), Q]))
        dCdm = QQ.dot(dRdm) + RR.dot(dQdm)
        dCds = QQ.dot(dRds) + RR.dot(dQds)
        dCdp = QQ.dot(dRdp) + RR.dot(dQdp)
        
        return M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds,  dMdp, dSdp, dCdp
   
    return M,S,C


#GPのコントローラー
def congp(nargout,policy, m, s):
    policy.hyp = policy.param['hyp'];
    policy.inputs = policy.param['inputs'];
    policy.targets = policy.param['targets'];
    
    policy.hyp[-2,:] = np.log(1);                 # set signal variance to 1
    policy.hyp[-1,:] = np.log(0.01);              # set noise standard dev to 0.01
    
    
#    % 2. Compute predicted control u inv(s)*covariance between input and control
    if nargout < 4:                                # if no derivatives are required
        [M, S, C] = gp2(policy, m, s);
        return M, S, C
    else:                                          #else compute derivatives too
        [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, dMdi, dSdi, dCdi, dMdt, dSdt, dCdt, dMdh, dSdh, dCdh] = gp2d(18,policy, m, s);
  
##  % 3. Set derivatives of non-free parameters to zero: signal and noise variance
        d = np.size(policy.inputs,1)        
        d2 = np.size(policy.hyp,0)
        dimU = np.size(policy.targets,1)
        sidx = np.atleast_2d(np.arange(d,d2)).T + np.atleast_2d(np.arange(0,dimU))*d2;#怪しい
        dMdh[:,np.reshape(sidx,[np.size(sidx),1],order = 'F')] = 0
        dSdh[:,np.reshape(sidx,[np.size(sidx),1],order = 'F')] = 0
        dCdh[:,np.reshape(sidx,[np.size(sidx),1],order = 'F')] = 0

##        % 4. Merge derivatives
        dMdp = np.hstack([dMdh,dMdi,dMdt])
        dSdp = np.hstack([dSdh,dSdi,dSdt])
        dCdp = np.hstack([dCdh,dCdi,dCdt])
        
        return M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, dMdp, dSdp, dCdp


def gSin(nargin,nargout,m, v, i, e):
    d = m.size
    I = i.size
    i = i.reshape([-1]).T;
    if nargin < 4:
        e = np.ones([I, 1])
    e = e.reshape([-1,1])
    
    mi = m[i]
    vi = v[np.ix_(i,i)]
    vii = np.atleast_2d(np.diag(vi)).T #short-hand notation
    M = e*np.exp(-vii/2)*np.sin(mi);      #mean
    
    lq = -(vii + vii.T)/2
    q = np.exp(lq)
    V = (np.exp(lq+vi)-q)*np.cos((mi - mi.T)) - (np.exp(lq-vi)-q)*np.cos((mi + mi.T))
    V = e.dot(e.T)*V/2# variance

    C = np.zeros([d,I])
    C[i,:] = np.diag((e*np.exp(-vii/2)*np.cos(mi))[:,0]); # inv(v) times cov

    if nargout > 3:                                          # compute derivatives?
        dVdm = np.zeros([I,I,d]);
        dCdm = np.zeros([d,I,d]); 
        dVdv = np.zeros([I,I,d,d]); 
        dCdv = np.zeros([d,I,d,d]);
        dMdm = C.T;
        U1 = -(np.exp(lq+vi)-q)*np.sin(mi- mi.T);
        U2 = (np.exp(lq-vi)-q)*np.sin(mi + mi.T);

        for j in range(I):
            u = np.zeros([I,1])
            u[j] = 1/2

            dVdm[:,:,i[j]] = e.dot(e.T)*(U1*(u-u.T) + U2*(u +u.T))  
            dVdv[j,j,i[j],i[j]] = np.exp(-vii[j]) * (1+(2*np.exp(-vii[j])-1)*np.cos(2*mi[j]))*e[j]*e[j]/2;
            for k in np.append(np.arange(j),np.arange(j+1,I)):
                dVdv[j,k,i[j],i[k]] = (np.exp(lq[j,k]+vi[j,k])*np.cos(mi[j]-mi[k]) + np.exp(lq[j,k]-vi[j,k])*np.cos(mi[j]+mi[k]))*e[j]*e[k]/2;
                dVdv[j,k,i[j],i[j]] = -V[j,k]/2; 
                dVdv[j,k,i[k],i[k]] = -V[j,k]/2; 

            dCdm[i[j],j,i[j]] = -M[j];
            dCdv[i[j],j,i[j],i[j]] = -C[i[j],j]/2;
        
        dMdv = np.transpose(dCdm,[1, 0, 2])/2;
  
        dMdv = np.reshape(dMdv,[I, d*d],order = 'F');
        dVdv = np.reshape(dVdv,[I*I, d*d],order = 'F');
        dVdm = np.reshape(dVdm,[I*I, d],order = 'F');
        dCdv = np.reshape(dCdv,[d*I, d*d],order = 'F'); 
        dCdm = np.reshape(dCdm,[d*I, d],order = 'F');
        
        return M, V, C, dMdm, dVdm, dCdm, dMdv, dVdv, dCdv

    return M,V,C
    

def gSat(nargin,nargout,m, v, i, e):
    d = m.size
    I = i.size
    i = i.reshape([-1,1]).T;
    if nargin < 4:
        e = np.ones([1, I])
    e = e.reshape([-1,1]).T
    
    P = np.vstack([np.eye(d), 3*np.eye(d)]);
    
    ma = P.dot(m)
    madm = P;
    va = P.dot(v).dot(P.T)
    vadv = np.kron(P,P)
    va = (va+va.T)/2;
    
    [M2, S2, C2, Mdma, Sdma, Cdma, Mdva, Sdva, Cdva] = gSin(4,9,ma, va, np.hstack([i, d+i]), np.hstack([9*e, e])/8);

    P = np.hstack([np.eye(I), np.eye(I)])
    Q = np.hstack([np.eye(d), 3*np.eye(d)])
    M = P.dot(M2);                                                # mean
    S = P.dot(S2).dot(P.T)
    S = (S+S.T)/2;                                    # variance
    C = Q.dot(C2).dot(P.T)                                    # inv(v) times input-output cov

    if nargout > 3:                                      # derivatives if required
        dMdm = P.dot(Mdma).dot(madm);         
        dMdv = P.dot(Mdva).dot(vadv);
        
        dSdm = np.kron(P,P).dot(Sdma).dot(madm) 
        dSdv = np.kron(P,P).dot(Sdva).dot(vadv)
        
        dCdm = np.kron(P,Q).dot(Cdma).dot(madm) 
        dCdv = np.kron(P,Q).dot(Cdva).dot(vadv)
                
        return M, S, C, dMdm, dSdm, dCdm, dMdv, dSdv, dCdv
    
    return M,S,C