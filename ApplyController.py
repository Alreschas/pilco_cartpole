# -*- coding: utf-8 -*-

import numpy as np
import scipy

from utility import gTrig

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


class Par:
    def __init__(self):
        self.dt = 0
        self.delay = 0
        self.tau = 0

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
    simi = np.array(simi,dtype='int')
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
        s = np.array([x[i,dyno]]).T
        sa = gTrig(s, np.zeros([len(s),len(s)]), angi)[0]
        s = np.vstack([s,sa])

        x[i,-2*nA:] = s[-2*nA:].T
        
        if(policy.fcn is 0):
            u[i,:] = policy.maxU*(2*np.random.rand(1,nU)-1)
        else:
            u[i,:] = policy.fcn[0](1,policy.fcn[1][0],policy.fcn[1][1],policy, s[poli], np.zeros([np.size(poli),np.size(poli)]))

        latent[i,:] = np.hstack([state, u[i:i+1,:]]);

        _next[0,odei] = simulate(state[0,odei], u[i,:], plant);
        if subi != []:
            _next[0,subi] = plant.subplant(state, u[i,:]);
        state[0,simi] = _next[0,simi]; 
        state[0,augi] = plant.augment(state);
#        x[i+1,simi] = state[0,simi] + np.random.randn(np.size(simi)).dot(np.linalg.cholesky(plant.noise).T);
        x[i+1,simi] = state[0,simi] + np.ones([np.size(simi)]).dot(np.linalg.cholesky(plant.noise).T);#for debug
        x[i+1,augi] = plant.augment(x[i+1,:]);
        
        if nargout > 2:
            L[0,i] = (cost.fcn(cost,state[:,dyno].T,np.zeros([len(dyno),len(dyno)])))[0]
        

    y = x[1:H+1,0:nX]
    x = np.hstack([x[0:H,:], u[0:H,:]]) 
    latent[H, 0:nX] = state
    latent = latent[0:H+1,:];
    L = L[0,0:H];

    return x, y, L, latent

  
    
def gaussian(m, S, *n):
    if n is ():
        n = 1
    else:
        n = n[0]

#    tmp = np.random.randn(np.size(S,1),n)
    tmp = np.ones([np.size(S,1),n])#for debug
    x = m[...,:] + np.linalg.cholesky(S).dot(tmp);

    return x

def applyController(HH,policy,plant,cost,mu0,S0):
    [xx, yy, realCost, latent] = rollout(gaussian(mu0, S0),policy,HH,plant,cost,4)
    return xx,yy,realCost,latent
