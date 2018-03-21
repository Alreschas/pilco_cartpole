# -*- coding: utf-8 -*-

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import scipy.integrate
from drawer import Drawer
import sys
from numpy.linalg import solve

odei = np.array([1, 2, 3, 4]);          # varibles for the ode solver
augi = np.array([]);                    # variables to be augmented
dyno = np.array([1, 2, 3, 4])           # variables to be predicted (and known to loss)
angi = np.array([4])                    # angle variables
dyni = np.array([1, 2, 3, 5, 6])        # variables that serve as inputs to the dynamics GP
poli = np.array([1, 2, 3, 5, 6])        # variables that serve as inputs to the policy
difi = np.array([1, 2, 3, 4])           # variables that are learned via differences

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
    class Param:
        def ___init__(self):
            self.hyp = 0
            self.targets = 0
            self.inputs = 0
    def __init__(self):
        self.maxU=np.array([10])
        self.p = self.Param()
        self.fcn = 0

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
        self.hyp = 0
        
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
    
    SW2 = (np.eye(D)+SW)
    SW2[SW2==0] = 1 #for zero divide
    iSpW = W/SW2;

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
        SW2 = (np.eye(D)+2*SW)
        SW2[SW2==0] = 1 #for zero divide
        i2SpW = W/SW2;
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
    M[0:D0,0] = m
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
    if(plant.augment is 0):
        plant.augment = lambda x:[]
        augi = []
    else:
        augi = plant.augi
    
    if(plant.subplant is 0):
        plant.subplant = lambda x,y:[]
        subi = []
    else:
        subi = plant.subi
    odei = plant.odei
    poli = plant.poli
    dyno = plant.dyno
    angi = plant.angi
    simi = np.sort(np.hstack([odei, subi]));
    simi = np.array(simi-1,dtype='int')
    nX = len(simi)+len(augi)
    nU = len(policy.maxU)
    nA = len(angi);
    
    state = np.zeros([len(simi)])
    state[simi] = np.copy(start[:,0])
    state[augi] = plant.augment(state)
    
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
            print("error")
#            u[i,:] = policy.maxU.*(2*rand(1,nU)-1)
        latent[i,:] = np.hstack([state, u[i,:]]);

        _next[0,odei-1] = simulate(state[odei-1], u[i,:], plant);
        if subi != []:
            _next[0,subi-1] = plant.subplant(state, u[i,:]);
        state[simi] = _next[0,simi]; 
        state[augi] = plant.augment(state);
#        x[i+1,simi] = state[simi] + np.random.randn(np.size(simi)).dot(np.linalg.cholesky(plant.noise).T);
        x[i+1,simi] = state[simi] + np.ones([np.size(simi)]).dot(np.linalg.cholesky(plant.noise).T);#for debug
        x[i+1,augi] = plant.augment(x[i+1,:]);
        
        if nargout > 2:
            L[0,i] = (cost.fcn(cost,state[dyno-1].T,np.zeros([len(dyno),len(dyno)])))[0]
        

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
    v = np.reshape(s,[len(s),1],order="F")
    return v

def rewrap(s, v):    # map elements of v (vector) onto s (any type)
    if(np.size(v) < np.size(s)):
        sys.stderr.write('The vector for conversion contains too few elements')
    s = np.reshape(v[0:np.size(s)], [np.size(s),1],order="F");
    v = v[np.size(s):]
    
    return s,v
           

def f(nargout,*varargin):
    if nargout == 0:
        f.p = varargin 
        f.F = f.p[0]
    else:
        [s,v] = rewrap(f.p[1], varargin[0])
        [fx, dfx] = f.F(s, f.p[2][0], f.p[2][1], f.p[2][2], f.p[2][3])
        dfx = unwrap(dfx);
        return fx,dfx
#        return

def LBFGS(x0, fx0, dfx0, p):
    p.SIG = 0.5  #default for line search quality
#n = length(x0); k = 0; ok = 1; x = x0; fx = fx0; bs = -1/p.MSR;
#if isfield(p, 'mem'), m = p.mem; else m = min(100, n); end    % set memory size
#a = zeros(1, m); t = zeros(n, m); y = zeros(n, m);            % allocate memory
#i = p.length < 0;                                 % initialize resource counter
#while i < abs(p.length)
#  q = dfx0;
#  for j = rem(k-1:-1:max(0,k-m),m)+1
#    a(j) = t(:,j)'*q/rho(j); q = q-a(j)*y(:,j);
#  end
#  if k == 0, r = -q/(q'*q); else r = -t(:,j)'*y(:,j)/(y(:,j)'*y(:,j))*q; end
#  for j = rem(max(0,k-m):k-1,m)+1
#    r = r-t(:,j)*(a(j)+y(:,j)'*r/rho(j));
#  end
#  s = r'*dfx0; if s >= 0, r = -dfx0; s = r'*dfx0; k = 0; ok = 0; end
#  b = bs/min(bs,s/p.MSR);              % suitable initial step size (usually 1)
#  b = min(b,1/norm(r));                    % limit step size in parameter space
#  b = max(b,1e-7/norm(r));
#  if isnan(r) | isinf(r)                                % if nonsense direction
#    i = -i;                                              % try steepest or stop
#  else
#    [x, b, fx0, dfx, i] = lineSearch(x0, fx0, dfx0, r, s, b, i, p); 
#  end
#  if i < 0                                              % if line search failed
#    i = -i; if ok, ok = 0; k = 0; else break; end        % try steepest or stop
#  else
#    j = rem(k,m)+1; t(:,j) = x-x0; y(:,j) = dfx-dfx0; rho(j) = t(:,j)'*y(:,j);
#    ok = 1; k = k+1; bs = b*s;
#  end
#  x0 = x; dfx0 = dfx; fx = [fx; fx0];                  % replace and add values
#end
#
#function [x, a, fx, df, i] = lineSearch(x0, f0, df0, d, s, a, i, p)
#if p.length < 0, LIMIT = min(p.MFEPLS, -i-p.length); else LIMIT = p.MFEPLS; end
#p0.x = 0.0; p0.f = f0; p0.df = df0; p0.s = s; p1 = p0;         % init p0 and p1
#j = 0; p3.x = a; wp(p0, p.SIG, 0);         % set step & Wolfe-Powell conditions
#if p.verbosity > 2
#  A = [-a a]/5; nd = norm(d); ah = ahandles(p);
#  hold(ah(2),'off'); plot(ah(2),0, f0, 'k+'); hold(ah(2),'on'); plot(ah(2),nd*A, f0+s*A, 'k-');
#  xlabel(ah(2),'distance in line search direction'); ylabel(ah(2),'function value');
#end
#while 1                               % keep extrapolating as long as necessary
#  ok = 0; 
#  while ~ok && j < LIMIT
#    try           % try, catch and bisect to safeguard extrapolation evaluation
#      j = j+1; [p3.f p3.df] = f(x0+p3.x*d); p3.s = p3.df'*d; ok = 1; 
#      if isnan(p3.f+p3.s) || isinf(p3.f+p3.s)
#        error('Objective function returned Inf or NaN','');
#      end;
#    catch
#      if p.verbosity > 1, printf('\n'); warning(lasterr); end % warn or silence
#      p3.x = (p1.x+p3.x)/2; ok = 0; p3.f = NaN;  p3.s = NaN;% bisect, and retry
#    end
#  end
#  if p.verbosity > 2
#    ah = ahandles(p); hold(ah(2),'on');
#    plot(ah(2),nd*p3.x, p3.f, 'b+'); plot(ah(2),nd*(p3.x+A), p3.f+p3.s*A, 'b-'); drawnow
#  end
#  if wp(p3) || j >= LIMIT, break; end                                    % done?
#  p0 = p1; p1 = p3;                                  % move points back one unit
#  p3.x = p0.x + minCubic(p1.x-p0.x, p1.f-p0.f, p0.s, p1.s, 1);    % cubic extrap
#end
#while 1                                % keep interpolating as long as necessary
#  if isnan(p3.f+p3.s) || isinf(p3.f+p3.s); p2 = p1; break; end % if final extrap failed
#  if p1.f > p3.f, p2 = p3; else p2 = p1; end           % make p2 the best so far
#  if wp(p2) > 1 || j >= LIMIT, break; end                                % done?
#  p2.x = p1.x + minCubic(p3.x-p1.x, p3.f-p1.f, p1.s, p3.s, 0);    % cubic interp
#  ok = 0; 
#  while ~ok && j < LIMIT;   % until function successfully evaluated or j = LIMIT
#    try                                     % try to evaluate objective function
#       j = j+1; [p2.f p2.df] = f(x0+p2.x*d); p2.s = p2.df'*d; ok = 1;
#       if isnan(p2.f+p2.s) || isinf(p2.f+p2.s)
#           error('Objective function returned Inf or NaN','');
#       end
#    catch                             % failed to successfully evaluate function
#       if p.verbosity > 1, printf('\n'); warning(lasterr); end % warn or silence
#       p2.x = (p1.x+p2.x)/2; ok = 0; if LIMIT == j; p2 = p1; end
#    end
#  end
#  if p.verbosity > 2
#    ah = ahandles(p); hold(ah(2),'on');
#    plot(ah(2),nd*p2.x, p2.f, 'r+'); plot(ah(2),nd*(p2.x+A), p2.f+p2.s*A, 'r'); drawnow
#  end
#  if wp(p2) > -1 && p2.s > 0 || wp(p2) < -1, p3 = p2; else p1 = p2; end % bracket
#end
#x = x0+p2.x*d; fx = p2.f; df = p2.df; a = p2.x;        % return the value found
#if p.length < 0, i = i+j; else i = i+1; end % count func evals or line searches
#if p.verbosity, printf('%s %6i;  value %4.6e\r', p.S, i, fx); end 
#if wp(p2) < 2, i = -i; end                                   % indicate faliure 
#if p.verbosity > 2
#  ah = ahandles(p); hold(ah(1),'on'); hold(ah(2),'on');
#  if i>0, plot(ah(2),norm(d)*p2.x, fx, 'go'); end
#  plot(ah(1),abs(i), fx, '+'); drawnow;
#end



def minimize(X, F, p, *varargin):
    param = Paramaters()
    param.length = p
    if param.length > 0:
        param.S = 'linesearch #'
    else:
        param.S = 'function evaluation #'
    print(param.S)
    x = unwrap(X)

    if len(x) > 1000:
        param.method = "LBFGS"
    else:
        param.method = "BFGS"

    param.verbosity = 1
    param.MFEPLS = 10
    param.MSR = 100
    
    f(0,F, X, varargin)
    [fx,dfx] = f(2,x)
    
    [x, fX, i, p] = param.method(x, fx, dfx, p)
#    [fx, dfx] = f(2,x)
    
    return

def train(gpmodel, dump, _iter = [[-500, -1000]]):
    curb = Curb()
    D = np.size(gpmodel.inputs,1);
    covfunc = [covSum, [covSEard, covNoise]]
    E = np.size(gpmodel.targets,1);
    curb.snr = 1000
    curb.ls = 100
    curb.std = np.atleast_2d(np.std(gpmodel.inputs,axis=0,ddof=1));# standard deviation ddof = 1


    if(gpmodel.hyp == 0):
        gpmodel.hyp = np.zeros([D+2,E])
        nlml = np.zeros([1,E]);

        lh = np.matlib.repmat(np.hstack([np.log(curb.std),[[0, -1]]]).T,1,E)
        lh[D,:] = np.log(np.std(gpmodel.targets,axis = 0,ddof=1))
        lh[D+1,:] = np.log(np.std(gpmodel.targets,axis=0,ddof=1)/10)       
    else:
        lh = gpmodel.hyp;
        
    print("Train hyper-parameters of full GP ...")
    for i in range(1):
        minimize(lh[:,i], hypCurb, _iter[0,0], covfunc, gpmodel.inputs, gpmodel.targets[:,i], curb);
    

          
    
def trainDynModel():
    Du = len(policy.maxU)
    Da = len(plant.angi) # no. of ctrl and angles
    xaug = np.hstack([x[:,dyno-1], x[:,-Du-2*Da:-Du]])# x augmented with angles
    dynmodel.inputs = np.hstack([xaug[:,dyni-1], x[:,-Du:]])
    
    dynmodel.targets = y[:,dyno-1];
    dynmodel.targets[:,difi-1] = dynmodel.targets[:,difi-1] - x[:,dyno[difi-1]-1];
    
    train(dynmodel,plant, trainOpt)


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
policy.p.inputs = gaussian(mm[poli-1], ss[poli-1,:][:,poli-1], nc).T
#policy.p.targets = 0.1*np.random.randn(nc, len(policy.maxU))
policy.p.targets = 0.1*np.ones([nc,len(policy.maxU)])#for debug
policy.p.hyp = np.log(np.array([[1, 1, 1, 0.7, 0.7, 1, 0.01]])).T;
cost.gamma = 1
cost.p = 0.5
cost.width = np.array([0.25])
cost.expl = 0.0
cost.angle = plant.angi
cost.target = np.array([[0,0,0,np.pi]]).T
cost.fcn = loss_cp

dynmodel.induce = np.zeros([300,0,1])

trainOpt = np.array([[300, 500]]);

np.set_printoptions(linewidth=200)

x = 0
y = 0
fantasy.mean = [[]] * N
fantasy.std = [[]] * N
realCost = [[]]*N;
latent = [[]];
M =  [[]]*N;
Sigma =  [[]]*N;


for jj in range(J):
    [xx, yy, realCost[jj], latent[jj]] = rollout(gaussian(mu0, S0),policy,H,plant,cost,4)
    if x == 0:
        x = np.empty((0,np.size(xx,1)))
        y = np.empty((0,np.size(yy,1)))

    x = np.vstack([x, xx])
    y = np.vstack([y, yy])
    
#drawer.main(latent)

mu0Sim = mu0[dyno-1]
S0Sim = S0[np.ix_(dyno-1,dyno-1)]

for j in range(1):
    print(j)
    trainDynModel()

print("end")