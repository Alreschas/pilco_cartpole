# -*- coding: utf-8 -*-

import numpy as np
import numpy.matlib
import scipy.optimize


from utility import unwrap,rewrap


def f1(nargout,*varargin):
    if nargout == 0:
        f1.p = varargin 
        f1.F = f1.p[0]
    else:
        s = rewrap(f1.p[1], varargin[0])
        [fx, dfx] = f1.F(s, f1.p[2][0], f1.p[2][1], f1.p[2][2], f1.p[2][3])
        dfx = unwrap(dfx);
        return fx,dfx

class GaussianProcess:
    def __init__(self,lh0,*varargin):
        self.F = hypCurb
        f1(0,self.F, lh0, varargin)
    def targetFunc(self,lh):
        [fx,dfx] = f1(2,lh)
        return fx[0]
    def targetFunc_dev(self,lh):
        [fx,dfx] = f1(2,lh)
        return dfx[:,0]

def solve_chol(A,B):
    x = np.linalg.solve(A,np.linalg.solve(A.T,B))
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
            W = np.linalg.solve(L.T,(np.linalg.solve(L,np.eye(n))))-alpha.dot(alpha.T);
            for i in range(len(out2)):
                out2[i]=np.sum(W*covfunc[0](4,1,covfunc[1], logtheta, x, i))/2
            return out1,out2
    #ok

    
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
 
    print('error')
    return 0,0

class Curb:
    def __init__(self):
        ""

def train(gpmodel, dump):
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
        print('GP learn:',i)
        gp_own = GaussianProcess(lh[:,i],covfunc, gpmodel.inputs, gpmodel.targets[:,i], curb)
        result = scipy.optimize.minimize(gp_own.targetFunc,lh[:,i],jac=gp_own.targetFunc_dev,method='BFGS')
        gpmodel.hyp[:,i] = result['x']
        train.nlml[i] = result['fun']
    

    
    [N, D] = gpmodel.inputs.shape; 
    [M, uD, uE] = gpmodel.induce.shape;
    if M >= N:
        print("Because of too few training expamples, we don't need FITC")
        return    # if too few training examples, we don't need FITC
 

#ダイナミクスの学習
def trainDynModel(dynmodel,policy,plant,x,y):

    dyno = plant.dyno 
    angi = plant.angi 
    dyni = plant.dyni
    difi = plant.difi
    
    Du = len(policy.maxU)
    Da = len(plant.angi) # no. of ctrl and angles
    xaug = np.hstack([x[:,dyno], x[:,-Du-2*Da:-Du]])# x augmented with angles
    
    dynmodel.inputs = np.hstack([xaug[:,dyni], x[:,-Du:]])
    dynmodel.targets = y[:,dyno];
    dynmodel.targets[:,difi] = dynmodel.targets[:,difi] - x[:,dyno[difi]];
    
    train(dynmodel,plant)
    
    Xh = dynmodel.hyp;

    print('Learned noise std: ' ,np.exp(Xh[-1,:]))
    print('SNRs             : ' ,np.exp(Xh[-2,:]-Xh[-1,:]))
