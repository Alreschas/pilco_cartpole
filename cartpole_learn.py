# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict

from PolicyEvaluation import loss_cp
from PolicyLearning import propagated
from PolicyLearning import learnPolicy
from DynamicsLearning import trainDynModel

from GP import gp1d
from GP import init_gp

from Controller import conCat, congp, gSat
from ApplyController import gaussian
from ApplyController import applyController
from ApplyController import rollout

from drawer import Drawer
from dynamics import dynamics_cp
from utility import gTrig


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
        self.noise = np.diag(np.ones([1, 4])[0] * 0.01**2)
        self.dynamics = 0
        self.delay = 0
        self.tau = 0
        self.dynamics = dynamics_cp
        self.prop = propagated


class Policy:
    def __init__(self, plant, mu0, S0):
        self.maxU = np.array([10])
        self.param = OrderedDict()
        self.fcn = 0
        self.nigp = 0

        (mm, ss, cc) = gTrig(mu0, S0, plant.angi, 3)

        mm = np.vstack([mu0, mm])
        cc = S0.dot(cc)
        ss = np.vstack([np.hstack([S0, cc]), np.hstack([cc.T, ss])])

        self.param['hyp'] = np.log(np.array([[1, 1, 1, 0.7, 0.7, 1, 0.01]])).T
        self.param['inputs'] = gaussian(mm[poli], ss[poli,:][:,poli], nc).T
#        self.param['inputs'] = 1 * np.sin(np.arange(1, 51).reshape([10, 5], order='F'))
#        self.param['targets'] = 0.1*np.ones([nc,np.size(self.maxU)])#for debug
#        self.param['targets'] = 1 * np.sin(np.arange(1, 1 + nc * len(self.maxU)).reshape([nc, len(self.maxU)], order='F'))  # for debug
        self.param['targets'] = 0.1*np.random.randn(nc, np.size(self.maxU))


class Cost:
    def __init__(self):
        self.fcn = 0                       # cost function
        self.gamma = 0                            # discount factor
        self.p = 0                              # length of pendulum
        self.width = 0                         # cost function width
        self.expl = 0.0                          # exploration parameter (UCB)
        self.angle = 0                   # index of angle (for cost function)
        self.target = 0                 # target state

        self.gamma = 1
        self.p = 0.5
        self.width = np.array([0.25])
        self.expl = 0.0
        self.target = np.array([[0, 0, 0, np.pi]]).T
        self.fcn = loss_cp


class Dynmodel:
    def __init__(self):
        self.hyp = np.empty(0)
        self.sub = 0
        self.nigp = 0

        self.induce = np.zeros([300, 0, 1])
        self.fcn = gp1d


class Fantasy:
    def __init__(self, N):
        self.mean = [[]] * N
        self.std = [[]] * N


np.random.seed(1)
np.set_printoptions(linewidth=200)
np.set_printoptions(precision=15)

odei = np.array([0, 1, 2, 3])          # varibles for the ode solver
augi = np.array([])                    # variables to be augmented
dyno = np.array([0, 1, 2, 3])           # variables to be predicted (and known to loss)
angi = np.array([4])                    # angle variables
dyni = np.array([0, 1, 2, 4, 5])        # variables that serve as inputs to the dynamics GP
poli = np.array([0, 1, 2, 4, 5])        # variables that serve as inputs to the policy
difi = np.array([0, 1, 2, 3])           # variables that are learned via differences

dt = 0.10                              # [s] sampling time
T = 4.0                                # [s] initial prediction horizon time
H = int(np.ceil(T / dt))                         # prediction steps (optimization horizon)
mu0 = np.atleast_2d([0, 0, 0, 0]).T    # initial state mean
S0 = np.diag([0.1, 0.1, 0.1, 0.1])**2   # initial state covariance
N = 4                                 # number controller optimizations
J = 1                                  # initial J trajectories of length H
K = 1                                  # no. of initial states for which we optimize
nc = 10                                # number of controller basis functions


drawer = Drawer()
plant = Plant()
policy = Policy(plant, mu0, S0)
cost = Cost()
cost.angle = plant.angi
dynmodel = Dynmodel()
fantasy = Fantasy(N)

realCost = [[]] * (N + 1)
latent = [[]] * N
M = [[]] * N
Sigma = [[]] * N

# drawer.main()


def learn(j):
    if j >= N:
        return
    print('episode:', j)
    global dynmodel, policy, plant, x, y, cost, fantasy, H, mu0Sim, S0Sim, M, Sigma
    trainDynModel(dynmodel, policy, plant, x, y)

    [M[j], Sigma[j], fantasy.mean[j], fantasy.std[j]] = learnPolicy(policy, dynmodel, plant, cost, H, mu0Sim, S0Sim)

    [xx, yy, realCost[j + J], latent[j]] = applyController(H, policy, plant, cost, mu0, S0)
    x = np.vstack([x, xx])
    y = np.vstack([y, yy])

    drawer.setLatent(latent[j])


drawer.setIdleFunc(learn)

x = 0
y = 0
for jj in range(J):
    [xx, yy, realCost[jj], latent[jj]] = rollout(gaussian(mu0, S0), policy, H, plant, cost, 4)
    if x == 0:
        x = np.empty((0, np.size(xx, 1)))
        y = np.empty((0, np.size(yy, 1)))

    x = np.vstack([x, xx])
    y = np.vstack([y, yy])


policy.fcn = [conCat, [congp, gSat]]


mu0Sim = mu0[dyno]
S0Sim = S0[np.ix_(dyno, dyno)]

init_gp()

drawer.setLatent(latent[0])
drawer.main()

#for j in range(N):
#    learn(j)

print("end")
