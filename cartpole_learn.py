# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

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
    ee = np.reshape(np.atleast_2d([e, e]),(2*I,1));
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
    tmp = np.ones([np.size(S,1),n])
    x = m[...,:] + np.linalg.cholesky(S).T.dot(tmp);

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
    tmp2 = np.ones([1,np.size(simi)])
    x[0,simi] = start.T + tmp2.dot(np.linalg.cholesky(plant.noise))#np.random.randn(1,np.size(simi)).dot(np.linalg.cholesky(plant.noise));
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
        tmp = np.ones([np.size(simi)])
    #    x[i+1,simi] = state[simi] + np.random.randn(np.size(simi)).dot(np.linalg.cholesky(plant.noise));
        x[i+1,simi] = state[simi] + tmp.dot(np.linalg.cholesky(plant.noise));
        x[i+1,augi] = plant.augment(x[i+1,:]);
        
        if nargout > 2:
            L[0,i] = (cost.fcn(cost,state[dyno-1].T,np.zeros([len(dyno),len(dyno)])))[0]
        

    y = x[1:H+1,0:nX]
    x = np.hstack([x[0:H,:], u[0:H,:]]) 
    latent[H, 0:nX] = state
    latent = latent[0:H+1,:];
    L = L[0,0:H];

    return


np.random.seed(1)

plant = Plant()
policy = Policy()
cost = Cost()
dynmodel = Dynmodel()
opt = Opt()
fantasy = Fantasy()

(mm, ss, cc) = gTrig(mu0, S0, plant.angi,3);

mm = np.vstack([mu0, mm])
cc = S0.dot(cc)
ss = np.vstack([np.hstack([S0,cc]),np.hstack([cc.T,ss])])
policy.p.inputs = gaussian(mm[poli-1], ss[poli-1,:][:,poli-1], nc).T
#target = 0.1*np.random.randn(nc, len(policy.maxU))
policy.p.targets = 0.1*np.ones([nc,len(policy.maxU)])
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


x = []
y = []
fantasy.mean = [[]] * N
fantasy.std = [[]] * N
realCost = [[]]*N;
M =  [[]]*N;
Sigma =  [[]]*N;


for jj in range(J):
    rollout(gaussian(mu0, S0),policy,H,plant,cost,4)
