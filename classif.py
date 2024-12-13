###############################################################################
# MODULES
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import numpy.random as rnd
from sklearn.linear_model import LogisticRegression
###############################################################################

################################################################################
# PARAMETERS
################################################################################
# Dimension and sample size
p=2
n=600
# Proportion of sample from classes 0, 1, and outliers
p0 = 3/6
p1 = 2/6
pout = 1/6
# Examples of means/covariances of classes 0, 1 and outliers
mu0 = np.array([-2,-2])
mu1 = np.array([2,2])
muout = np.array([-8,-8])
Sigma_ex1 = np.eye(p)
Sigma_ex2 = np.array([[5, 0.1],
                      [1, 0.5]])
Sigma_ex3 = np.array([[0.5, 1],
                      [1, 5]])
Sigma0 = Sigma_ex1
Sigma1 = Sigma_ex1
Sigmaout = Sigma_ex1
# Regularization coefficient
lamb = 0
################################################################################

################################################################################
# DATA/LABELS GENERATION
################################################################################
# Sample sizes
n0 = int(n*p0)
n1 = int(n*p1)
nout = int(n*pout)
# Data and labels
mu0_mat = mu0.reshape((p,1))@np.ones((1,n0))
mu1_mat = mu1.reshape((p,1))@np.ones((1,n1))
x0 = np.zeros((p,n0+nout))
x0[:,0:n0] = mu0_mat + la.sqrtm(Sigma0)@rnd.randn(p,n0)
x1 = mu1_mat + la.sqrtm(Sigma1)@rnd.randn(p,n1)
if nout > 0:
  muout_mat = muout.reshape((p,1))@np.ones((1,nout))
  x0[:,n0:n0+nout] = muout_mat + la.sqrtm(Sigmaout)@rnd.randn(p,nout)
y = np.concatenate((-np.ones(n0+nout),np.ones(n1)))
X = np.ones((n,p+1))
for i in np.arange(n):
     X[0:n0+nout,1:p+1] = x0.T
     X[n0+nout:n,1:p+1] = x1.T
print(np.shape(x0))
################################################################################

################################################################################
# CODE
################################################################################
# Trouver les hyperplans 
################################################################################

rlog = LogisticRegression()
rlog.fit(X,y)
coeff = rlog.coef_[0]
X_pred=np.linspace(-10,6,20)
Y_pred=-(coeff[1]/coeff[2])*X_pred-(coeff[0]/coeff[2])
################################################################################
# PLOTS
################################################################################
fig,ax = plt.subplots()
ax.plot(X_pred,Y_pred)
ax.plot(x0[0,:],x0[1,:],'xb',label='Class 0')
ax.plot(x1[0,:],x1[1,:],'xr',label="Class 1")
ax.legend(loc = "upper left")
plt.ylim(np.min(x0[1,:])-0.5,np.max(x1[1,:])+0.5)
plt.show()