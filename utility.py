# -*- coding: utf-8 -*-
import numpy as np
import copy

from collections import OrderedDict

#一列に展開
def unwrap(s):
    v = np.atleast_2d(np.array([[]]))
    if isinstance(s,OrderedDict):
        n = list(s.values())
        for i in n:
#            print(v)
            tmp = np.reshape(i,[np.size(i),1],order='F')
#            print(np.reshape(i,[np.size(i),1],order='F'))
            v = np.append(v,tmp)
        v = np.atleast_2d(v).T
    else:
        v = np.reshape(s,[np.size(s),1],order="F")
    return v

#vを、辞書型など,sの型に戻す
def rewrap(s, v):    # map elements of v (vector) onto s (any type)
    if(np.size(v) < np.size(s)):
        sys.stderr.write('The vector for conversion contains too few elements')
    rets = copy.deepcopy(s)
    retv = copy.deepcopy(v)

    if isinstance(s,np.ndarray):
        rets = np.reshape(v[0:np.size(s)], [np.size(s),1],order='F');
        retv = v[np.size(rets):]
    else:    
        st = 0
        idx = 0
        for i in s:
            tgt = s[i]
            tmp = v[st:st+np.size(tgt)]
            st = st + np.size(tgt)
            ret = tmp.reshape(tgt.shape,order = 'F')
            rets[i] = ret    
    return rets
           

def gTrig(m,v,i,nargout=3,*e):
    M = np.zeros([2,1])
    V = np.zeros([2,2])    

    
    d = len(m)
    I = len(i)
    Ic = 2*np.arange(1,I+1)-1;
    Is = Ic-1;
    if(e is ()):
        e = np.ones([I,1])
#    else:
#        e = e(:);
    ee = np.reshape(np.atleast_2d([e, e]),(2*I,1),order="F");
    mi = np.zeros(I)
    mi[0] = m[i-1]
    

    vi = v[i-1,i-1]
    vii = np.zeros(I)
    vii = np.diag(vi)
    
    M[Is,0] = e*np.exp(-vii/2)*np.sin(mi)
    M[Ic,0] = e*np.exp(-vii/2)*np.cos(mi)    # mean
    
    lq = -(vii[...,:] + vii.T)/2
    q = np.exp(lq)
    
    U1 = (np.exp(lq+vi)-q)*np.sin(mi[...,:] - mi.T)
    U2 = (np.exp(lq-vi)-q)*np.sin(mi[...,:] + mi.T)
    U3 = (np.exp(lq+vi)-q)*np.cos(mi[...,:] - mi.T)
    U4 = (np.exp(lq-vi)-q)*np.cos(mi[...,:] + mi.T)
    

    V[Is,Is] = U3 - U4; 
    V[Ic,Ic] = U3 + U4;
    V[Is,Ic] = U1 + U2; 
    V[Ic,Is] = V[Is,Ic].T
    V = ee.dot(ee.T)*V/2 # variance
    
    C = np.zeros([d,2*I]);
    C[i-1,Is] = np.diag(M[Ic]);
    C[i-1,Ic] = np.diag(-M[Is]);
    
    
    if nargout > 3:  # compute derivatives?
        dVdm = np.zeros([2*I,2*I,d])
        dCdm = np.zeros([d,2*I,d])
        dVdv = np.zeros([2*I,2*I,d,d])
        dCdv = np.zeros([d,2*I,d,d])
        dMdm = C.T;
        
        for j in range(0,I):
            u = np.zeros([I,1]);
            u[j] = 1/2;
            
            dVdm[Is,Is,i[j]-1] = e.dot(e.T)*(-U1*(u-u.T)+U2*(u+u.T))
            dVdm[Ic,Ic,i[j]-1] = e.dot(e.T)*(-U1*(u-u.T)-U2*(u+u.T))
            dVdm[Is,Ic,i[j]-1] = e.dot(e.T)*(U3*(u-u.T) +U4*(u+u.T))
            dVdm[Ic,Is,i[j]-1] = dVdm[Is,Ic,i[j]-1].T 
            dVdv[Is[j],Is[j],i[j]-1,i[j]-1] = np.exp(-vii[j,0]) * (1+(2*np.exp(-vii[j,0])-1) * np.cos(2*mi[j])) * e[j,0] * e[j,0]/2
            dVdv[Ic[j],Ic[j],i[j]-1,i[j]-1] = np.exp(-vii[j,0]) * (1-(2*np.exp(-vii[j,0])-1) * np.cos(2*mi[j])) * e[j,0] * e[j,0]/2;
            dVdv[Is[j],Ic[j],i[j]-1,i[j]-1] = np.exp(-vii[j,0]) * (1-2*np.exp(-vii[j,0]))    * np.sin(2*mi[j])  * e[j,0] * e[j,0]/2;
            dVdv[Ic[j],Is[j],i[j]-1,i[j]-1] = dVdv[Is[j],Ic[j],i[j]-1,i[j]-1];

            tmp = []
            if(j-1 > 0):                
                tmp = np.hstack([np.arange(0,j-1), np.arange(j,I)])
                
            for k in tmp:
                print("error")
                dVdv[Is[j],Is[k],i[j]-1,i[k]-1] = (np.exp(lq[j,k]+vi[j,k])*np.cos(mi[j]-mi[k]) + exp(lq[j,k]-vi[j,k])*cos(mi[j]+mi[k]))*e[j]*e[k]/2;
#                dVdv[Is[j],Is[k],i[j]-1,i[j]-1] = -V(Is[j],Is[k])/2; 
#                dVdv[Is[j],Is[k],i[k]-1,i[k]-1] = -V(Is(j),Is(k))/2; 
#                dVdv[Ic[j],Ic[k],i[j]-1,i[k]-1] = (exp(lq(j,k)+vi(j,k)).*cos(mi(j)-mi(k)) - exp(lq(j,k)-vi(j,k)).*cos(mi(j)+mi(k)))*e(j)*e(k)/2;
#                dVdv[Ic[j],Ic[k],i[j]-1,i[j]-1] = -V(Ic(j),Ic(k))/2; 
#                dVdv[Ic[j],Ic[k],i[k]-1,i[k]-1] = -V(Ic(j),Ic(k))/2; 
#                dVdv[Ic[j],Is[k],i[j]-1,i[k]-1] = -(exp(lq(j,k)+vi(j,k)).*sin(mi(j)-mi(k)) + exp(lq(j,k)-vi(j,k)).*sin(mi(j)+mi(k)))*e(j)*e(k)/2;
#                dVdv[Ic[j],Is[k],i[j]-1,i[j]-1] = -V(Ic(j),Is(k))/2; 
#                dVdv[Ic[j],Is[k],i[k]-1,i[k]-1] = -V(Ic(j),Is(k))/2; 
#                dVdv[Is[j],Ic[k],i[j]-1,i[k]-1] = (exp(lq(j,k)+vi(j,k)).*sin(mi(j)-mi(k)) - exp(lq(j,k)-vi(j,k)).*sin(mi(j)+mi(k)))*e(j)*e(k)/2;
#                dVdv[Is[j],Ic[k],i[j]-1,i[j]-1] = -V(Is(j),Ic(k))/2; 
#                dVdv[Is[j],Ic[k],i[k]-1,i[k]-1] = -V(Is(j),Ic(k))/2; 
#
            dCdm[i[j]-1,Is[j],i[j]-1] = -M[Is[j]]
            dCdm[i[j]-1,Ic[j],i[j]-1] = -M[Ic[j]];
            dCdv[i[j]-1,Is[j],i[j]-1,i[j]-1] = -C[i[j]-1,Is[j]]/2;
            dCdv[i[j]-1,Ic[j],i[j]-1,i[j]-1] = -C[i[j]-1,Ic[j]]/2;

#        dMdv = permute(dCdm,[2 1 3])/2;
        dMdv = np.transpose(dCdm, [1, 0, 2])/2
        
        dMdv = np.reshape(dMdv,[2*I, d*d],order="F")
        dVdv = np.reshape(dVdv,[4*I*I, d*d],order="F")
        dVdm = np.reshape(dVdm,[4*I*I, d],order="F")
        dCdv = np.reshape(dCdv,[d*2*I, d*d],order="F")
        dCdm = np.reshape(dCdm,[d*2*I, d],order="F")
        

        return M, V, C, dMdm, dVdm, dCdm, dMdv, dVdv, dCdv
    else:
        return M, V, C
          