# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict
#from scipy import linalg
from PolicyEvaluation import loss_cp
from PolicyLearning import propagated
from PolicyLearning import learnPolicy
from DynamicsLearning import trainDynModel

from GP import gp0d
from GP import init_gp

from Controller import conCat
from ApplyController import applyController
from ApplyController import rollout

from drawer import Drawer
from cartpole import dynamics_cp
from utility import gaussian
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
#        self.noise = np.eye(4) * (0.01**2) #ダイナミクスに対するノイズ
        self.noise = np.eye(4) * 0.0001**2  # ダイナミクスに対するノイズなしの場合
        self.dynamics = dynamics_cp
        self.prop = propagated


class Policy:
    def __init__(self, plant, mu0, S0, nc):
        self.maxU = np.array([10])
        self.param = OrderedDict()
        self.fcn = 0
        self.nigp = 0

        (mm, ss, cc) = gTrig(mu0, S0, plant.angi, 3)

        mm = np.vstack([mu0, mm])
        cc = S0.dot(cc)
        ss = np.vstack([np.hstack([S0, cc]), np.hstack([cc.T, ss])])

        self.param['hyp'] = np.log(np.array([[1, 1, 1, 0.7, 0.7, 1, 0.01]])).T
        self.param['inputs'] = gaussian(mm[poli], ss[poli, :][:, poli], nc).T
#        self.param['inputs'] = 1 * np.sin(np.arange(1, 51).reshape([10, 5], order='F'))
#        self.param['targets'] = 0.1*np.ones([nc,np.size(self.maxU)])#for debug
#        self.param['targets'] = 1 * np.sin(np.arange(1, 1 + nc * len(self.maxU)).reshape([nc, len(self.maxU)], order='F'))  # for debug
        self.param['targets'] = 0.1 * np.random.randn(nc, np.size(self.maxU))


class Cost:
    def __init__(self):
        self.expl = 0.0      # 探索パラメータ(UCB)
        self.gamma = 1  # コスト減衰率
        self.p = 0.5  # 振り子の長さ
        self.width = np.array([0.25])
        self.expl = 0.0  # 探索パラメータ
        self.target = np.array([[0, 0, 0, np.pi]]).T  # 目標位置
        self.angle = 0                   # 角度パラメータのインデックス
        self.fcn = loss_cp  # コスト関数


class Dynmodel:
    def __init__(self):
        self.hyp = np.empty(0)
        self.nigp = 0
        self.fcn = gp0d


class Fantasy:
    def __init__(self, N):
        self.mean = [[]] * N
        self.std = [[]] * N


np.random.seed(0)
np.set_printoptions(linewidth=200)
np.set_printoptions(precision=15)

odei = np.array([0, 1, 2, 3])           # varibles for the ode solver
dyno = np.array([0, 1, 2, 3])           # ロボットの状態変数へのインデックス　variables to be predicted (and known to loss)
angi = np.array([3])                    # 角度変数のインデックス
dyni = np.array([0, 1, 2, 4, 5])        # variables that serve as inputs to the dynamics GP
poli = np.array([0, 1, 2, 4, 5])        # variables that serve as inputs to the policy
difi = np.array([0, 1, 2, 3])           # variables that are learned via differences

dt = 0.10                              # サンプリングタイム[s]
T = 4.0                                # 1エピソードの時間[s]
H = int(np.ceil(T / dt))               # １エピソードのステップ数
mu0 = np.atleast_2d([0, 0, 0, 0]).T    # 初期状態の平均値
# S0 = np.eye(4) * 0.1**2  # 初期状態の分散
S0 = np.eye(4) * 0.0001**2  # ノイズ無しの場合
N = 5                                  # ロールアウト回数(学習回数+1)
J = 1                                  # 最初に何回ロールアウトするか
nc = 10                                # コントローラの基底関数の数


drawer = Drawer()
plant = Plant()

cost = Cost()
cost.angle = plant.angi

policy = Policy(plant, mu0, S0, nc)
dynmodel = Dynmodel()

fantasy = Fantasy(N)

realCost = [[]] * N
latent = [[]] * N
M = [[]] * N
Sigma = [[]] * N

init_gp()

fin = False


# 学習処理メイン
def learn(j):
    global dynmodel, policy, plant, x, y, cost, fantasy, H, mu0, S0, M, Sigma, fin

    # 所定の回数を超えたら、終了フラグを立てる
    if j >= N:
        fin = True
        drawer.reset()
        j = 0

    # 学習終了後は、プレイバック
    if(fin == True):
        drawer.setLatent(latent[j])
        return

    print('episode:', j + 1)

    # ダイナミクスの学習
    trainDynModel(dynmodel, policy, plant, x, y)

    # ポリシーの学習
    [M[j], Sigma[j], fantasy.mean[j], fantasy.std[j]] = learnPolicy(policy, dynmodel, plant, cost, H, mu0, S0)

    # 学習したポリシーでシミュレーション
    [xx, yy, realCost[j], latent[j]] = applyController(H, policy, plant, cost, mu0, S0)

    # シミュレーション結果を、スタックに積んでいく
    x = np.vstack([x, xx])
    y = np.vstack([y, yy])

    # ロボットの動作結果設定
    drawer.setLatent(latent[j])


# 初回実行(乱数で実行)
x = 0
y = 0
for jj in range(J):
    [xx, yy, realCost[jj], latent[jj]] = rollout(gaussian(mu0, S0), policy, H, plant, cost)
    if x == 0:
        x = np.empty((0, np.size(xx, 1)))
        y = np.empty((0, np.size(yy, 1)))

    x = np.vstack([x, xx])
    y = np.vstack([y, yy])

# 制御入力を計算する関数を設定
policy.fcn = conCat


# 描画開始
drawer.setIdleFunc(learn)  # 描画していないときに実行する関数
drawer.setLatent(latent[0])  # ロボットの動作結果設定
drawer.main()  # 描画開始

# 描画したく無いときは、こっちを使う
# for j in range(N):
#    learn(j)

print("end")
