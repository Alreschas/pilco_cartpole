# -*- coding: utf-8 -*-

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.optimize
from drawer import Drawer
import sys
from numpy.linalg import solve
from collections import OrderedDict
import copy

odei = np.array([1, 2, 3, 4]);          # varibles for the ode solver
augi = np.array([]);                    # variables to be augmented
dyno = np.array([1, 2, 3, 4])           # variables to be predicted (and known to loss)
angi = np.array([4])                    # angle variables
dyni = np.array([0, 1, 2, 4, 5])        # variables that serve as inputs to the dynamics GP
poli = np.array([0, 1, 2, 4, 5])        # variables that serve as inputs to the policy
difi = np.array([0, 1, 2, 3])           # variables that are learned via differences

dt = 0.10;                              # [s] sampling time
T = 4.0;                                # [s] initial prediction horizon time
H = int(np.ceil(T/dt));                         # prediction steps (optimization horizon)
mu0 = np.atleast_2d([0, 0, 0, 0]).T;    # initial state mean
S0 = np.diag([0.1, 0.1, 0.1, 0.1])**2   # initial state covariance
N = 15;                                 # number controller optimizations
J = 1;                                  # initial J trajectories of length H
K = 1;                                  # no. of initial states for which we optimize
nc = 10;                                # number of controller basis functions

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
            
    
def gaussian(m, S, *n):
    if n is ():
        n = 1
    else:
        n = n[0]

#    tmp = np.random.randn(np.size(S,1),n)
    tmp = np.ones([np.size(S,1),n])#for debug
    x = m[...,:] + np.linalg.cholesky(S).dot(tmp);

    return x
        

class Plant:
    def __init__(self):
        self.dt = dt
        self.odei = odei
        self.dyno = dyno
        self.angi = angi
        self.dyni = dyni
        self.poli = poli
        self.difi = difi
        self.augment = 0
        self.subplant = 0
        self.noise = np.diag(np.ones([1,4])[0]*0.01**2);
        self.dynamics = 0
        self.delay = 0
        self.tau = 0
        self.dynamics = dynamics_cp 

class Policy:
    def __init__(self):
        self.maxU=np.array([10])
        self.param = OrderedDict()
        self.fcn = 0
        self.nigp = 0

class Cost:
    def __init__(self):
        self.fcn = 0;                       # cost function
        self.gamma = 0;                            # discount factor
        self.p = 0;                              # length of pendulum
        self.width = 0;                         # cost function width
        self.expl =  0.0;                          # exploration parameter (UCB)
        self.angle = 0;                   # index of angle (for cost function)
        self.target = 0;                 # target state

class Dynmodel:
    def __init__(self):
        self.induce = 0
        self.hyp = np.empty(0)
        self.sub = 0
        self.nigp = 0
        
class Opt:
    def __init__(self):
        self.length = 150
        self.MFEPLS = 30
        self.verbosity = 1

class Fantasy:
    def __init__(self):
        self.mean = 0
        self.std = 0

class Par:
    def __init__(self):
        self.dt = 0
        self.delay = 0
        self.tau = 0

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


def zoh(f, t, par): # **************************** zero-order hold
    d = par.delay;

    if d==0:
        u = f;
    else:
        e = d/100
        t0=t-(d-e/2)
        if t < d - e/2:
            u=f[0];
        elif t < d+e/2:
            u=(1-t0/e)*f(1) + t0/e*f(2);    # prevents ODE stiffness
        else:
            u=f[1];
    return u


def dynamics_cp(t,z,f):
    l = 0.5;  # [m]      length of pendulum
    m = 0.5;  # [kg]     mass of pendulum
    M = 0.5;  # [kg]     mass of cart
    b = 0.1;  # [N/m/s]  coefficient of friction between cart and ground
    g = 9.82; # [m/s^2]  acceleration of gravity
    
    if f != 0:
        dz = np.zeros([4,1])
        dz[0] = z[1];
        dz[1] = ( 2*m*l*z[2]**2*np.sin(z[3]) + 3*m*g*np.sin(z[3])*np.cos(z[3]) + 4*f[0](t) - 4*b*z[1] )/( 4*(M+m)-3*m*np.cos(z[3])**2 );
        dz[2] = (-3*m*l*z[2]**2*np.sin(z[3])*np.cos(z[3]) - 6*(M+m)*g*np.sin(z[3]) - 6*(f[0](t)-b*z[1])*np.cos(z[3]) )/( 4*l*(m+M)-3*m*l*np.cos(z[3])**2 );
        dz[3] = z[2];
    else:
        dz = (M+m)*z[1]^2/2 + 1/6*m*l^2*z[2]^2 + m*l*(z[1]*z[2]-g)*np.cos(z[3])/2;

    return dz

def simulate(x0, f, plant):
    par = Par()
    x0 = x0[:]
    f = f[:] 
    nU = len(f)
    dt = plant.dt
    dynamics = plant.dynamics
    delay = plant.delay
    if plant.tau is 0:
        tau = dt
    else:
        tau = plant.tau
    if delay is 0:
        x0s = x0
        U = f
        _id = 0;

    par.delay = delay
    par.dt = dt
    par.tau = tau
    u0 = [[]]*nU

    for j in range(nU):
        u0[j] =lambda t:zoh(U[j],t,par)
    
    solver = scipy.integrate.ode(dynamics).set_integrator('dopri5',rtol=1e-12,atol=1e-12)
    solver.set_initial_value(x0s)
    solver.set_f_params(u0)
    
    while solver.successful() and solver.t < dt:
        solver.integrate(solver.t+dt/2)
    y = solver.y

    udt = np.zeros([nU,1])
    for j in range(nU):
        udt[j] = u0[j](dt)
        
    if _id==0:
        _next =  y;                         # return augmented state
    elif _id==1:
        _next = np.vstack([y, udt]);
    else:
        _next = np.vstack([y, f, udt]);

    return _next


def rollout(start, policy, H, plant, cost,nargout):

    plant.augment = lambda x:[]
    augi = []
    
    plant.subplant = lambda x,y:[]
    subi = []

    odei = plant.odei
    poli = plant.poli
    dyno = plant.dyno
    angi = plant.angi
    simi = np.sort(np.hstack([odei, subi]));
    simi = np.array(simi-1,dtype='int')
    nX = len(simi)+len(augi)
    nU = len(policy.maxU)
    nA = len(angi);
    
    state = np.zeros([1,len(simi)])
    state[0,simi] = np.copy(start[:,0])
    state[0,augi] = plant.augment(state)
    
    x = np.zeros([H+1, nX+2*nA]);

#    x[0,simi] = start.T + np.random.randn(1,np.size(simi)).dot(np.linalg.cholesky(plant.noise).T);
    x[0,simi] = start.T + np.ones([1,np.size(simi)]).dot(np.linalg.cholesky(plant.noise).T) #  for debug
    x[0,augi] = plant.augment(x[1,:]);
    
    u = np.zeros([H, nU])
    latent = np.zeros([H+1, np.size(state)+nU])
    y = np.zeros([H, nX])
    L = np.zeros([1, H]);
    _next = np.zeros([1,len(simi)]); 
  
    for i in range(H):
        s = np.array([x[i,dyno-1]]).T
        sa = gTrig(s, np.zeros([len(s),len(s)]), angi)[0]
        s = np.vstack([s,sa])

        x[i,-2*nA:] = s[-2*nA:].T
        
        if(policy.fcn is 0):
            u[i,:] = policy.maxU*(2*np.random.rand(1,nU)-1)
        else:
            u[i,:] = policy.fcn[0](1,policy.fcn[1][0],policy.fcn[1][1],policy, s[poli], np.zeros([np.size(poli),np.size(poli)]))

        latent[i,:] = np.hstack([state, u[i:i+1,:]]);

        _next[0,odei-1] = simulate(state[0,odei-1], u[i,:], plant);
        if subi != []:
            _next[0,subi-1] = plant.subplant(state, u[i,:]);
        state[0,simi] = _next[0,simi]; 
        state[0,augi] = plant.augment(state);
#        x[i+1,simi] = state[simi] + np.random.randn(np.size(simi)).dot(np.linalg.cholesky(plant.noise).T);
        x[i+1,simi] = state[0,simi] + np.ones([np.size(simi)]).dot(np.linalg.cholesky(plant.noise).T);#for debug
        x[i+1,augi] = plant.augment(x[i+1,:]);
        
        if nargout > 2:
            L[0,i] = (cost.fcn(cost,state[:,dyno-1].T,np.zeros([len(dyno),len(dyno)])))[0]
        

    y = x[1:H+1,0:nX]
    x = np.hstack([x[0:H,:], u[0:H,:]]) 
    latent[H, 0:nX] = state
    latent = latent[0:H+1,:];
    L = L[0,0:H];

    return x, y, L, latent

class Curb:
    def __init__(self):
        ""

class Paramaters:
    def __init__(self):
        ""

def solve_chol(A,B):
    x = solve(A,solve(A.T,B))
    return x

def gpr(nargout,logtheta, covfunc, x, y, *xstar):
    
    if xstar == () :
        nargin = 4
    [n,D] = x.shape
    
    y = np.atleast_2d(y).T
    
    if(eval('D+2') != np.size(logtheta, 0)):
        print('Error: Number of parameters do not agree with covariance function')

    K = covfunc[0](3,1,covfunc[1],logtheta,x,0)
    L = np.linalg.cholesky(K);
    alpha = solve_chol(L.T,y);
    
    
    if nargin == 4:
        out1 = 0.5*y.T.dot(alpha) + np.sum(np.log(np.diag(L))) + 0.5*n*np.log(2*np.pi)
        
        if( nargout==2 ):
            out2 = np.zeros(np.shape(logtheta));
            W = solve(L.T,(solve(L,np.eye(n))))-alpha.dot(alpha.T);
            for i in range(len(out2)):
                out2[i]=np.sum(W*covfunc[0](4,1,covfunc[1], logtheta, x, i))/2
            return out1,out2
    #ok

    
    return 0,0


def sq_dist(nargin,a, b, Q):

    if(nargin == 1 or b == []):
        b = a; 
    [D, n] = a.shape; 
    [d, m] = b.shape;
    if d != D:
        print('Error: column lengths must agree.')
    if nargin < 3:
        C = np.zeros([n,m])
        for d in range(D):
            C = C + (np.matlib.repmat(np.atleast_2d(b[d,:]), n, 1) - np.matlib.repmat(np.atleast_2d(a[d,:]).T, 1, m))**2;
    else:
        if [n, m] == Q.shape:
            C = np.zeros([D,1]);
            for d in range(D):
                C[d] = np.sum(np.sum((np.matlib.repmat(b[d,:], n, 1) - np.matlib.repmat(a[d,:].T, 1, m))**2*Q));
        else:
            print('Third argument has wrong size.');
    
    return C
    
def covNoise(nargin,nargout,logtheta, x, z):

    s2 = np.exp(2*logtheta);#noise variance

    if nargin == 2:# compute covariance matrix
        A = s2*np.eye(np.size(x,0));
        return A
    elif nargout == 2:   # compute test set covariances
        A = s2
        B = 0                               #zeros cross covariance by independence
        return A,B
    else:                                   #compute derivative matrix
        A = 2*s2*np.eye(np.size(x,0))
        return A
    
def covSEard(nargin,nargout,loghyper, x, z):
    [n, D] = x.shape;
    ell = np.exp(loghyper[:D]);  # characteristic length scale
    sf2 = np.exp(2*loghyper[D]); # signal variance

    if nargin == 2:
        covSEard.K = sf2*np.exp(-sq_dist(1,np.diag(1./ell[:,0]).dot(x.T),[],0)/2)
        A = covSEard.K;
        return A
    elif nargout == 2:
        A = sf2*np.ones([np.size(z,0),1]);
        B = sf2*np.exp(-sq_dist(2,np.diag(1./ell[:,0]).dot(x.T),np.diag(1./ell[:,0]).dot(z.T),0)/2);
        return A,B
    else:
        if(np.any(np.array(covSEard.K.shape) != n)):
            covSEard.K = sf2*np.exp(-sq_dist(1,np.diag(1./ell[:,0]).dot(x.T),[],0)/2)
        if z <= D-1: # length scale parameters
            A = covSEard.K*sq_dist(1,np.atleast_2d(x[:,z])/ell[z],[],0);
            return A
        else:# magnitude parameter
            A = 2*covSEard.K;
            covSEard.K = [];
            return A
        
    return 0,0

def covSum(nargin,nargout,covfunc, logtheta, x, z):
    
    j = ['D+1','1']
    [n, D] = x.shape
    v = np.array([[]]);              # v vector indicates to which covariance parameters belong
    for i in range(len(covfunc)):
        v = np.hstack([v, np.matlib.repmat(i, 1, eval(j[i]))])

    if(nargin == 3):
        A = np.zeros([n, n]);
        for i in range(len(covfunc)):      #iteration over summand functions
            f = covfunc[i]
            A = A + f(2,1,np.atleast_2d(logtheta[(v==i).T]).T, x,0)            
        return A
    elif(nargin == 4):
        if nargout == 2:          #compute test set cavariances
            A = np.zeros([np.size(z,0),1]);
            B = np.zeros([np.size(x,0),np.size(z,0)]);   # allocate space
            for i in range(len(covfunc)):
                f = covfunc[i];
                [AA, BB] = f(3,2,logtheta[(v==i).T], x, z);  # compute test covariances
                A = A + AA 
                B = B + BB                                  # and accumulate
        else:                     # compute derivative matrices
            i = int(v[0,z]);# which covariance function
            j = np.sum(v[0,:z] == i);             #which parameter in that covariance
            f = covfunc[i];
            A = f(3,1,logtheta[(v==i).T], x, j);  #compute derivative
            return A
 
    return 0,0

##input
#lh:対数ハイパーパラメータ
#covfunc:共分散行列計算関数
#x:トレーニングインプット
#y:トレーニングターゲット
#curb:ペナルティー
##output
#周辺対数尤度
#周辺対数尤度の微分
def hypCurb(lh, covfunc, x, y, curb):
    p = 30;  #penalty power
    D = np.size(x,1);

    if np.size(lh,0) == 3*D+2: 
        li = np.arange(2*D)
        sfi = np.arange(2*D,3*D+1) # 1D and DD terms
    elif np.size(lh,0) == 2*D+1:
        li = np.arange(D)
        sfi = np.arange(D,2*D)   # Just 1D terms
    elif np.size(lh,0) == D+2:
        li = np.arange(D)
        sfi = D      # Just DD terms
    else:
        print('Incorrect number of hyperparameters'); 

    ll = lh[li]
    lsf = lh[sfi]
    lsn = lh[-1];
        
    [f, df] = gpr(2,lh, covfunc, x, y);
    
    # 2) add penalties and change derivatives accordingly
    f = f + np.sum(((ll - np.log(curb.std.T))/np.log(curb.ls))**p);  # length-scales
    df[li] = df[li] + p*(ll - np.log(curb.std.T))**(p-1)/np.log(curb.ls)**p;

    f = f + np.sum(((lsf - lsn)/np.log(curb.snr))**p); # signal to noise ratio
    df[sfi] = df[sfi] + p*(lsf - lsn)**(p-1)/np.log(curb.snr)**p;
    df[-1] = df[-1] - p*np.sum((lsf - lsn)**(p-1)/np.log(curb.snr)**p);
    
    return f,df

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
           

def f(nargout,*varargin):
    if nargout == 0:
        f.p = varargin 
        f.F = f.p[0]
    else:
        s = rewrap(f.p[1], varargin[0])
        [fx, dfx] = f.F(s, f.p[2][0], f.p[2][1], f.p[2][2], f.p[2][3])
        dfx = unwrap(dfx);
        return fx,dfx

def f2(nargout,*varargin):
    if nargout == 0:
        f.p = varargin 
        f.F = f.p[0]
    else:
        s = rewrap(f.p[1], varargin[0])

        [fx,dfx] = f.F(s, f.p[2][0],f.p[2][1],f.p[2][2],f.p[2][3],f.p[2][4],f.p[2][5],f.p[2][6])

        return fx,dfx

#####own
class GaussianProcess:
    def __init__(self,lh0,*varargin):
        self.F = hypCurb
        f(0,self.F, lh0, varargin)
    def targetFunc(self,lh):
        [fx,dfx] = f(2,lh)
        return fx[0]
    def targetFunc_dev(self,lh):
        [fx,dfx] = f(2,lh)
        return dfx[:,0]
    
class PolicyOptimizer:
    def __init__(self,x0, *varargin):
        self.F = value
        f2(0,self.F, x0, varargin)
        self.dfx = np.empty(0)
        self.fx = np.empty(0)
    def targetFunc(self,x):
        if(np.size(self.fx) == 0):
            [fx,dfx] = f2(2,x)
            self.dfx = unwrap(dfx)
        else:
            fx = np.copy(self.fx)
        self.fx = np.empty(0)
        return fx

    def targetFunc_dev(self,x):
        if(np.size(self.dfx) == 0):
            [fx,dfx] = f2(2,x)
            dfx = unwrap(dfx)
            self.fx = np.copy(fx)
        else:
            dfx = np.copy(self.dfx)

        self.dfx = np.empty(0)

        return dfx[:,0]
#        return dfx[:,0]
    
########

#X:初期値
#F:最小化する関数
#p:パラメータ
#def minimize(X, F, p, *varargin):
#    param = Paramaters()
#    param.length = p
#    if param.length > 0:
#        param.S = 'linesearch #'
#    else:
#        param.S = 'function evaluation #'
#    print(param.S)
#    x = unwrap(X)
#
#    if len(x) > 1000:
#        param.method = LBFGS
#    else:
#        param.method = BFGS
#
#    param.verbosity = 1
#    param.MFEPLS = 10
#    param.MSR = 100
#    
#    f(0,F, X, varargin)
#    [fx,dfx] = f(2,x)
#    
#    [x, fX, i, p] = param.method(x, fx, dfx, p)
##    [fx, dfx] = f(2,x)
#    
#    return

def train(gpmodel, dump, _iter = [[-500, -1000]]):
    curb = Curb()
    D = np.size(gpmodel.inputs,1);
    covfunc = [covSum, [covSEard, covNoise]]
    E = np.size(gpmodel.targets,1);
    curb.snr = 1000
    curb.ls = 100
    curb.std = np.atleast_2d(np.std(gpmodel.inputs,axis=0,ddof=1));# standard deviation ddof = 1


    if(np.size(gpmodel.hyp) == 0):
        gpmodel.hyp = np.zeros([D+2,E])
        train.nlml = np.zeros([E]);

        lh = np.matlib.repmat(np.hstack([np.log(curb.std),[[0, -1]]]).T,1,E)
        lh[D,:] = np.log(np.std(gpmodel.targets,axis = 0,ddof=1))
        lh[D+1,:] = np.log(np.std(gpmodel.targets,axis=0,ddof=1)/10)       
    else:
        lh = gpmodel.hyp;
        
    print("Train hyper-parameters of full GP ...")
    
    for i in range(E):
        gp_own = GaussianProcess(lh[:,i],covfunc, gpmodel.inputs, gpmodel.targets[:,i], curb)
        result = scipy.optimize.minimize(gp_own.targetFunc,lh[:,i],jac=gp_own.targetFunc_dev,method='BFGS')
        gpmodel.hyp[:,i] = result['x']
        train.nlml[i] = result['fun']
    
    [N, D] = gpmodel.inputs.shape; 
    [M, uD, uE] = gpmodel.induce.shape;
    if M >= N:
        print("Because of too few training expamples, we don't need FITC")
        return    # if too few training examples, we don't need FITC
    
def fillIn(nargin,nargout,S,C,mdm,sdm,Cdm,mds,sds,Cds,Mdm,Sdm,Mds,Sds,Mdp,Sdp,dCdp,i,j,k,D):
    if k is ():
        return
    X = np.reshape(np.arange(D*D),[D, D],order='F') 
    XT = X.T;                         #vectorized indices
    I=0*X; I[np.ix_(i,i)]=1; ii=XT[(I==1).T].T;
    I=0*X; I[np.ix_(k,k)]=1; kk=XT[(I==1).T].T;
    I=0*X; I[np.ix_(j,i)]=1; ji=XT[(I==1).T].T;
    I=0*X; I[np.ix_(j,k)]=1; jk=XT[(I==1).T].T; kj=X[(I==1).T].T;

    Mdm[k,:]  = mdm.dot(Mdm[i,:]) + mds.dot(Sdm[ii,:])# chainrule
    Mds[k,:]  = mdm.dot(Mds[i,:]) + mds.dot(Sds[ii,:])
    Sdm[kk,:] = sdm.dot(Mdm[i,:]) + sds.dot(Sdm[ii,:])
    Sds[kk,:] = sdm.dot(Mds[i,:]) + sds.dot(Sds[ii,:])
    dCdm      = Cdm.dot(Mdm[i,:]) + Cds.dot(Sdm[ii,:])
    dCds      = Cdm.dot(Mds[i,:]) + Cds.dot(Sds[ii,:])
    
    if nargin < 19 and nargout > 5 :
        Mdp[k,:]  = mdm.dot(Mdp[i,:]) + mds.dot(Sdp[ii,:]);
        Sdp[kk,:] = sdm.dot(Mdp[i,:]) + sds.dot(Sdp[ii,:]);
        dCdp      = Cdm.dot(Mdp[i,:]) + Cds.dot(Sdp[ii,:]);
    elif nargout > 5:
        aa = np.size(k)        
        bb = aa**2
        cc = np.size(C);
        
        mdp = np.zeros([D,np.size(Mdp,1)])
        sdp = np.zeros([D*D,np.size(Mdp,1)])
        mdp[k,:]  = np.reshape(Mdp,[aa,-1],order = 'F')
        Mdp = mdp;
        
        sdp[kk,:] = np.reshape(Sdp,[bb,-1],order = 'F')
        Sdp = sdp;
        
        Cdp       = np.reshape(dCdp,[cc,-1],order = 'F')
        dCdp = Cdp;

        
    q = S[np.ix_(j,i)].dot(C)
    S[np.ix_(j,k)] = q;
    S[np.ix_(k,j)] = q.T # off-diagonal
    SS = np.kron(np.eye(np.size(k)),S[np.ix_(j,i)])
    CC = np.kron(C.T,np.eye(np.size(j)));
    Sdm[jk,:] = SS.dot(dCdm) + CC.dot(Sdm[ji,:]);
    Sdm[kj,:] = Sdm[jk,:];
    Sds[jk,:] = SS.dot(dCds) + CC.dot(Sds[ji,:]);
    Sds[kj,:] = Sds[jk,:];
    
    if nargout > 5:
        Sdp[jk,:] = SS.dot(dCdp) + CC.dot(Sdp[ji,:])
        Sdp[kj,:] = Sdp[jk,:];

    if(nargout ==5):
        return S, Mdm, Mds, Sdm, Sds
    if(nargout == 7):
        return S, Mdm, Mds, Sdm, Sds, Mdp, Sdp

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

            if 1/numpy.linalg.cond(R) < 1e-15:
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

    
def sliceModel(dynmodel,n,ii,D1,D2,D3): #separate sub-dynamics
    if (dynmodel.sub != 0):
        dyn = dynmodel.sub[n]
        do = dyn.dyno
        D = np.size(ii)+D1-D2
        if (dyn.dyni != []):di=dyn.dyni
        else: di=[];
        if (dyn.dynu != []):du=dyn.dynu
        else: du=[];
        if (dyn.dynj != []):dj=dyn.dynj
        else: dj=[];
        i = np.hstack([ii[di], D1+du, D2+dj])
        k = D2+do
        dyn.inputs = np.hstack([dynmodel.inputs[:,np.hstack([di, D+du])], dynmodel.target[:,dj]]); # inputs
        dyn.target = dynmodel.target[:,do];                                 #targets
    else:
        dyn = dynmodel;
        k = np.arange(D2,D3)
        i = ii;

    return dyn, i, k


def propagate(m, s, plant, dynmodel, policy):
    angi = plant.angi
    poli = plant.poli
    dyni = plant.dyni
    difi = plant.difi
    
    D0 = len(m);                #size of the input mean
    D1 = D0 + 2*len(angi);          #length after mapping all angles to sin/cos
    D2 = D1 + len(policy.maxU)     #length after computing control signal
    D3 = D2 + D0;                      #length after predicting
    M = np.zeros([D3,1])
    M[0:D0] = m
    S = np.zeros([D3,D3])
    S[0:D0,0:D0] = s   #init M and S
  
    i = np.arange(D0)
    j = np.arange(D0)
    k = np.arange(D0,D1);
    
    [M[k], S[np.ix_(k,k)], C] = gTrig(M[i], S[np.ix_(i,i)], angi,3);
    q = S[np.ix_(j,i)].dot(C)
    S[np.ix_(j,k)] = q;
    S[np.ix_(k,j)] = q.T;
    
    sn2 = np.exp(2*dynmodel.hyp[-1,:])
    sn2[difi] = sn2[difi]/2
    
    mm=np.zeros([D1,1])
    mm[i]=M[i]
    ss[np.ix_(i,i)]=S[np.ix_(i,i)]+np.diag(sn2);
    [mm[k], ss[np.ix_(k,k)], C] = gTrig(mm[i], ss[np.ix_(i,i)], angi,3); #noisy state measurement
    q = ss[np.ix_(j,i)].dot(C)
    ss[np.ix_(j,k)] = q;
    ss[np.ix_(k,j)] = q.T;
    
    i = poli
    j = np.arange(D1)
    k = np.arange(D1,D2)
    
    policy.fcn = [conCat,[congp,gSat]]
    [M[k], S[np.ix_(k,k)], C] = policy.fcn[0](3,policy.fcn[1][0],policy.fcn[1][1],policy, mm[i], ss[np.ix_(i,i)])
    q = S[np.ix_(j,i)].dot(C)
    S[np.ix_(j,k)] = q;
    S[np.ix_(k,j)] = q.T;

    ii = np.hstack([dyni, np.arange(D1,D2)])
    j = np.arange(D2);
    
    if(dynmodel.sub != 0):
        Nf = np.size(dynmodel.sub)
    else:
        Nf = 1

    for n in range(Nf):                               # potentially multiple dynamics models
        [dyn, i, k] = sliceModel(dynmodel,n,ii,D1,D2,D3);
        j = np.setdiff1d(j,k)

        [M[k], S[np.ix_(k,k)], C] = dyn.fcn(3,3,dyn,M[i],S[np.ix_(i,i)]);
  
        q = S[np.ix_(j,i)].dot(C)
        S[np.ix_(j,k)] = q;
        S[np.ix_(k,j)] = q.T;

        j = np.hstack([j, k]);

    P = np.hstack([np.zeros([D0,D2]), np.eye(D0)])
    P[np.ix_(difi,difi)] = np.eye(np.size(difi));

    Mnext = P.dot(M); 
    Snext = P.dot(S).dot(P.T);
    Snext = (Snext+Snext.T)/2;
    
    return Mnext, Snext

    
    
def propagated(m, s, plant, dynmodel, policy,nargout=8):
    
    if nargout <= 2:      #just predict, no derivatives
        [Mnext, Snext] = propagate(m, s, plant, dynmodel, policy);
        return Mnext,Snext

    
    angi = plant.angi
    poli = plant.poli
    dyni = plant.dyni
    difi = plant.difi
    
    D0 = len(m);                #size of the input mean
    D1 = D0 + 2*len(angi);          #length after mapping all angles to sin/cos
    D2 = D1 + len(policy.maxU)     #length after computing control signal
    D3 = D2 + D0;                      #length after predicting
    M = np.zeros([D3,1])
    M[0:D0] = m
    S = np.zeros([D3,D3])
    S[0:D0,0:D0] = s   #init M and S
    
    Mdm = np.vstack([np.eye(D0), np.zeros([D3-D0,D0])])
    Sdm = np.zeros([D3*D3,D0])
    Mds = np.zeros([D3,D0*D0])
    Sds = np.kron(Mdm,Mdm);
    X = np.reshape(np.arange(D3*D3),[D3, D3],order='F')
    XT = X.T
    Sds = (Sds + Sds[XT.reshape(np.size(XT),order='F'),:])/2;
    X = np.reshape(np.arange(D0*D0),[D0, D0])
    XT = X.T
    Sds = (Sds + Sds[:,XT.reshape(np.size(XT),order='F')])/2;
    
    
    i = np.arange(D0)
    j = np.arange(D0)
    k = np.arange(D0,D1);
    
    
    [M[k], S[np.ix_(k,k)], C, mdm, sdm, Cdm, mds, sds, Cds] = gTrig(M[i], S[np.ix_(i,i)], angi,9);
    
    [S, Mdm, Mds, Sdm, Sds] = fillIn(16,5,S,C,mdm,sdm,Cdm,mds,sds,Cds,Mdm,Sdm,Mds,Sds,[ ],[ ],[ ],i,j,k,D3)

    sn2 = np.exp(2*dynmodel.hyp[-1,:])
    sn2[difi] = sn2[difi]/2
    
    mm=np.zeros([D1,1])
    mm[i]=M[i]
    ss[np.ix_(i,i)]=S[np.ix_(i,i)]+np.diag(sn2);
    [mm[k], ss[np.ix_(k,k)], C] = gTrig(mm[i], ss[np.ix_(i,i)], angi,3); #noisy state measurement
    q = ss[np.ix_(j,i)].dot(C)
    ss[np.ix_(j,k)] = q;
    ss[np.ix_(k,j)] = q.T;
    
    i = poli
    j = np.arange(D1)
    k = np.arange(D1,D2)
    
    policy.fcn = [conCat,[congp,gSat]]
    [M[k], S[np.ix_(k,k)], C, mdm, sdm, Cdm, mds, sds, Cds, Mdp, Sdp, Cdp] = policy.fcn[0](12,policy.fcn[1][0],policy.fcn[1][1],policy, mm[i], ss[np.ix_(i,i)])

    [S, Mdm, Mds, Sdm, Sds, Mdp, Sdp] = fillIn(19,7,S,C,mdm,sdm,Cdm,mds,sds,Cds,Mdm,Sdm,Mds,Sds,Mdp,Sdp,Cdp,i,j,k,D3);
    
    ii = np.hstack([dyni, np.arange(D1,D2)])
    j = np.arange(D2);
    
    if(dynmodel.sub != 0):
        Nf = np.size(dynmodel.sub)
    else:
        Nf = 1
        
    for n in range(Nf):                               # potentially multiple dynamics models
        [dyn, i, k] = sliceModel(dynmodel,n,ii,D1,D2,D3);
        j = np.setdiff1d(j,k)
        
        [M[k], S[np.ix_(k,k)], C, mdm, sdm, Cdm, mds, sds, Cds] = dyn.fcn(3,9,dyn,M[i],S[np.ix_(i,i)]);
        
        [S, Mdm, Mds, Sdm, Sds, Mdp, Sdp] = fillIn(18,7,S,C,mdm,sdm,Cdm,mds,sds,Cds,Mdm,Sdm,Mds,Sds,Mdp,Sdp,[ ],i,j,k,D3);

        j = np.hstack([j, k]);
    
    P = np.hstack([np.zeros([D0,D2]), np.eye(D0)])
    P[np.ix_(difi,difi)] = np.eye(np.size(difi));

    Mnext = P.dot(M); 
    Snext = P.dot(S).dot(P.T);
    Snext = (Snext+Snext.T)/2;

    PP = np.kron(P,P);
    dMdm =  P.dot(Mdm)
    dMds =  P.dot(Mds)
    dMdp =  P.dot(Mdp)
    
    dSdm = PP.dot(Sdm)
    dSds = PP.dot(Sds)
    dSdp = PP.dot(Sdp)

    X = np.reshape(np.arange(D0*D0),[D0, D0],order = 'F');
    XT = X.T;     #symmetrize dS
    dSdm = (dSdm + dSdm[XT.reshape([-1]),:])/2;
    dMds = (dMds + dMds[:,XT.reshape([-1])])/2;
    
    dSds = (dSds + dSds[XT.reshape([-1]),:])/2;
    dSds = (dSds + dSds[:,XT.reshape([-1])])/2;
    
    dSdp = (dSdp + dSdp[XT.reshape([-1]),:])/2;
    
        
    return Mnext, Snext, dMdm, dSdm, dMds, dSds, dMdp, dSdp


def value(p, m0, S0, dynmodel, policy, plant, cost, H):
    
    policy.param = copy.deepcopy(p)
    p = unwrap(p);
    
    dp = 0*p;
    m = m0
    S = S0
    L = np.zeros([1,H]);
    

    
    dmOdp = np.zeros([np.size(m0,0), len(p)]);
    dSOdp = np.zeros([np.size(m0,0)* np.size(m0,0), len(p)]);
   
    for t in range(H): # for all time steps in horizon
        [m, S, dmdmO, dSdmO, dmdSO, dSdSO, dmdp, dSdp] = plant.prop(m, S, plant, dynmodel, policy) # get next state
    
        dmdp = dmdmO.dot(dmOdp) + dmdSO.dot(dSOdp) + dmdp;
        dSdp = dSdmO.dot(dmOdp) + dSdSO.dot(dSOdp) + dSdp;
    
        [L[0,t], dLdm, dLdS,tmp] = cost.fcn(cost, m, S);              #predictive cost

        L[0,t] = cost.gamma**t * L[0,t];                             # discount
        dp = dp + cost.gamma**t *( dLdm.reshape([-1,1],order = 'F').T.dot(dmdp) + dLdS.reshape([-1,1],order = 'F').T.dot(dSdp) ).T;

        dmOdp = np.copy(dmdp);
        dSOdp = np.copy(dSdp);                              #bookkeeping

    J = np.sum(L,keepdims = True)
    dJdp = rewrap(policy.param, dp);
    
    return J, dJdp
          
    
def trainDynModel():
    Du = len(policy.maxU)
    Da = len(plant.angi) # no. of ctrl and angles
    xaug = np.hstack([x[:,dyno-1], x[:,-Du-2*Da:-Du]])# x augmented with angles
    dynmodel.inputs = np.hstack([xaug[:,dyni], x[:,-Du:]])
    
    dynmodel.targets = y[:,dyno-1];
    dynmodel.targets[:,difi] = dynmodel.targets[:,difi] - x[:,dyno[difi]-1];
    
    train(dynmodel,plant, trainOpt)
    
    Xh = dynmodel.hyp;
    print('Learned noise std: ' ,np.exp(Xh[-1,:]))
    print('SNRs             : ' ,np.exp(Xh[-2,:]-Xh[-1,:]))


def pred(policy, plant, dynmodel, m, s, H):

    D = np.size(m)
    S = np.zeros([D,D,H+1])
    M = np.zeros([D,H+1])
    M[:,0:1] = m
    S[:,:,0] = s
    for i in range(H):
        [m, s] = plant.prop(m, s, plant, dynmodel, policy,2);
        M[:,i+1:i+2] = m[-D:,0:1]; 
        S[:,:,i+1] = s[-D:,-D:];
    return M,S


def calcCost(cost, M, S):

    H = np.size(M,1);           # horizon length
    L = np.zeros([1,H])
    SL = np.zeros([1,H])
    

#% for each time step, compute the expected cost and its variance
    for h in range(H):
        [L[0,h],d1,d2,SL[0,h]]  = cost.fcn(cost, M[:,h:h+1], S[:,:,h]);

    sL = np.sqrt(SL); 

    return L, sL

def learnPolicy(j):
    global M,Sigma
    opt.fh = 1;
#    unwrap(policy.p)


    X_fmt=policy.param
    X0 = unwrap(X_fmt)[:,0]
    
    po_own = PolicyOptimizer(X_fmt, mu0Sim, S0Sim, dynmodel, policy, plant, cost, H)
    
    result = scipy.optimize.minimize(po_own.targetFunc,X0,jac=po_own.targetFunc_dev,method='BFGS',options={'maxiter': 5, 'disp': False})
    
    policy.param = rewrap(policy.param,result['x'])
    fX3 = result['fun']
    
    [M[j],Sigma[j]] = pred(policy, plant, dynmodel, mu0Sim[:,0:1], S0Sim, H)

    [fantasy.mean[j], fantasy.std[j]] = calcCost(cost, M[j], Sigma[j])
    
def applyController(j):
    global realCost,latent
    HH = H
    [xx, yy, realCost[j+J], latent[j]] = rollout(gaussian(mu0, S0),policy,HH,plant,cost,4)
    

np.random.seed(1)

plant = Plant()
policy = Policy()
cost = Cost()
dynmodel = Dynmodel()
opt = Opt()
fantasy = Fantasy()
drawer = Drawer()

(mm, ss, cc) = gTrig(mu0, S0, plant.angi,3);

mm = np.vstack([mu0, mm])
cc = S0.dot(cc)
ss = np.vstack([np.hstack([S0,cc]),np.hstack([cc.T,ss])])
policy.param['hyp'] = np.log(np.array([[1, 1, 1, 0.7, 0.7, 1, 0.01]])).T;
policy.param['inputs'] = gaussian(mm[poli], ss[poli,:][:,poli], nc).T
policy.param['targets'] = 0.1*np.ones([nc,len(policy.maxU)])#for debug
#policy.p.targets = 0.1*np.random.randn(nc, len(policy.maxU))
cost.gamma = 1
cost.p = 0.5
cost.width = np.array([0.25])
cost.expl = 0.0
cost.angle = plant.angi
cost.target = np.array([[0,0,0,np.pi]]).T
cost.fcn = loss_cp

plant.prop = propagated

dynmodel.induce = np.zeros([300,0,1])
dynmodel.fcn = gp1d

trainOpt = np.array([[300, 500]]);

np.set_printoptions(linewidth=200)

x = 0
y = 0
fantasy.mean = [[]] * N
fantasy.std = [[]] * N
realCost = [[]]*(N+1);
latent = [[]]*N;
M =  [[]]*N;
Sigma =  [[]]*N;


for jj in range(J):
    [xx, yy, realCost[jj], latent[jj]] = rollout(gaussian(mu0, S0),policy,H,plant,cost,4)
    if x == 0:
        x = np.empty((0,np.size(xx,1)))
        y = np.empty((0,np.size(yy,1)))

    x = np.vstack([x, xx])
    y = np.vstack([y, yy])
    
drawer.main(latent)

mu0Sim = mu0[dyno-1]
S0Sim = S0[np.ix_(dyno-1,dyno-1)]

gp0.oldX = np.empty(0)
gp2.oldX = np.empty(0)
gp2d.oldX = np.empty(0)
gp0d.oldX = np.empty(0)

for j in range(N):
    print(j)
    trainDynModel()
    learnPolicy(j)
    applyController(j)

print("end")