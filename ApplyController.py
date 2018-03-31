# -*- coding: utf-8 -*-

import numpy as np
import scipy

from utility import gTrig
from utility import gaussian

#初期位置x0から、力fをかけて、時刻をdtだけ進める
def simulate(x0, f, plant):
    nU = np.size(f)
    dt = plant.dt
    dynamics = plant.dynamics


    #ODEソルバに、運動方程式などを設定
    u0 = [[]]*nU
    for j in range(nU): u0[j] = lambda t:f[j]  #一ステップの操作量は一定とする
    solver = scipy.integrate.ode(dynamics).set_integrator('dopri5',rtol=1e-12,atol=1e-12)
    solver.set_initial_value(x0)
    solver.set_f_params(u0)

    #指定した時刻までODEソルバの時刻を進める    
    while solver.successful() and solver.t < dt:
        solver.integrate(solver.t+dt/2)

    _next = solver.y;
    return _next


def rollout(start, policy, H, plant, cost):


    odei = plant.odei
    poli = plant.poli
    dyno = plant.dyno
    angi = plant.angi
    simi = np.sort(odei);
    simi = np.array(simi,dtype='int')
    nX = np.size(simi)
    nU = np.size(policy.maxU) #制御入力の数
    nA = np.size(angi); #角度変数の数
    
    state = np.zeros([1,np.size(simi)])
    state[0,simi] = np.copy(start[:,0])
    
    x = np.zeros([H+1, nX+2*nA]);

    #初期状態にノイズをのせる(コレスキー分解を使っているが、普通にmultivariate normalでOK)
    print(plant.noise)
    x[0,simi] = start.T + np.random.randn(1,np.size(simi)).dot(np.linalg.cholesky(plant.noise).T);

    
    u = np.zeros([H, nU])
    latent = np.zeros([H+1, np.size(state)+nU])
    y = np.zeros([H, nX])
    L = np.zeros([1, H]);
    _next = np.zeros([1,np.size(simi)]); 
  
    #ステップ時間繰り返し
    for i in range(H):
        s = np.array([x[i,dyno]]).T
        sa = gTrig(s, np.zeros([np.size(s),np.size(s)]), angi)[0]
        s = np.vstack([s,sa])

        x[i,-2*nA:] = s[-2*nA:].T
        
        if(policy.fcn is 0):
            #ポリシーが無い場合は、乱数で動かす
            u[i,:] = policy.maxU*(2*np.random.rand(1,nU)-1)
        else:
            #ポリシーに従って制御入力を決める
            u[i,:] = policy.fcn(1,policy, s[poli], np.zeros([np.size(poli),np.size(poli)]))

        latent[i,:] = np.hstack([state, u[i:i+1,:]]);

        #シミュレーションする
        _next[0,odei] = simulate(state[0,odei], u[i,:], plant);
        state[0,simi] = _next[0,simi]; 

        #シミュレーション結果に、ノイズをのせる
        x[i+1,simi] = state[0,simi] + np.random.randn(np.size(simi)).dot(np.linalg.cholesky(plant.noise).T);
        
        #コストの計算
        L[0,i] = (cost.fcn(cost,state[:,dyno].T,np.zeros([np.size(dyno),np.size(dyno)])))[0]
        

    y = x[1:H+1,0:nX]
    x = np.hstack([x[0:H,:], u[0:H,:]]) 
    latent[H, 0:nX] = state
    latent = latent[0:H+1,:];
    L = L[0,0:H];

    return x, y, L, latent


def applyController(HH,policy,plant,cost,mu0,S0):
    [xx, yy, realCost, latent] = rollout(gaussian(mu0, S0),policy,HH,plant,cost)
    return xx,yy,realCost,latent
