###############################################################################
# MODULES
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import numpy.random as rnd
from sklearn.linear_model import LinearRegression, Ridge
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
################################################################################

################################################################################
# CODE
################################################################################
# Trouver les hyperplans 
################################################################################
def Reg_lin(X,y): # Régression Linéaire 
    coeff=np.linalg.inv(np.dot(np.transpose(X),X))
    coeff=np.dot(coeff,np.transpose(X))
    coeff=np.dot(coeff,y)
    return coeff
  
def RIDGE(X, y, lamb):
    # Ajouter une colonne de 1 pour le terme d'interception
    X = np.vstack([np.ones(len(X)), X]).T

    # Créer et ajuster le modèle Ridge
    ridge = Ridge(alpha=lamb)
    ridge.fit(X, y)

    # Récupérer les coefficients
    intercept = ridge.intercept_
    coef = ridge.coef_

    return intercept, coef

def X_create(q,x):
    X=[[x[i]**j for j in range(0,q+1)] for i in range(len(x))]
    return X
  
def OLS(x, y): 
    len_data = len(x)
    X=X_create(1,x)

    x_data = x
    y_data = y

    # Use only one feature
    x_data = x_data[:, np.newaxis]

    # Split the data into training/testing sets
    x_train = x_data[:-len_data//2]
    x_test = x_data[-len_data//2:]

    # Split the targets into training/testing sets
    y_train = y_data[:-len_data//2]
    y_test = y_data[-len_data//2:]

    # Create linear regression object
    regr = LinearRegression()

    # Train the model using the training sets
    regr.fit(x_train, y_train)
    a=Reg_lin(X,y)

    # Make predictions using the testing set
    y_pred = regr.predict(x_test)

    return x_test, y_test, y_pred, a

# Distances aux hyperplans 
def distance_hyperplan(x, labels, intercept, coef):
  norme_beta = np.sqrt(np.sum(coef**2))
  distances = np.zeros(len(x))
  for i in range(len(x)):
    distances[i] = labels[i] * (coef[0] * x[i] + intercept) / norme_beta
  return distances

# Equation de l'hyperplan
def hyperplan(x, y, intercept, coef, point_min):
  m = - 1 / coef[0]
  b = point_min[1] - m * point_min[0]
  return m * x + b

def milieu(x, y):
  return (x + y) / 2

# Concaténation des données
labels = y
x = np.concatenate((x0[0,:], x1[0,:]))
y = np.concatenate((x0[1,:], x1[1,:]))

# Obtenir les coefficients OLS et Ridge pour chaque classe
x_ols, y_test, y_ols, a = OLS(x, y)
intercept_ridge, coef_ridge = RIDGE(x, y, lamb=1.0)
coef_ridge = coef_ridge[1:]
y_ridge = intercept_ridge + coef_ridge[0] * x
x_ridge = x

# On récupère le point le plus proche de l'hyperplan pour CHAQUE classe de données
distances_class0 = distance_hyperplan(x0[0,:], -np.ones(len(x0[0,:])), intercept_ridge, coef_ridge)
index_min_class0 = np.argmin(distances_class0)
distances_class1 = distance_hyperplan(x1[0,:], np.ones(len(x1[0,:])), intercept_ridge, coef_ridge)
index_min_class1 = np.argmin(distances_class1)

# On cherche ensuite le point situé au milieu de ces deux points pour déterminer l'équation de l'hyperplan
point_min = np.zeros(2)
point_min[0] = (x0[0,:][index_min_class0] + x0[0,:][index_min_class1]) / 2
point_min[1] = (x1[0,:][index_min_class0] + x1[0,:][index_min_class1]) / 2

hyperplan_equation = hyperplan(x, y, intercept_ridge, coef_ridge, point_min)

################################################################################
# PLOTS
################################################################################
fig,ax = plt.subplots()
ax.plot(x0[0,:],x0[1,:],'xb',label='Class 0')
ax.plot(x1[0,:],x1[1,:],'xr',label="Class 1")
plt.plot(x_ridge, y_ridge, color='red', label='Ridge')
#plt.plot(x_ols, y_ols, color="blue", label="OLS")
plt.plot(x, hyperplan_equation, color="green", label="Hyperplan")
ax.legend(loc = "upper left")
plt.show()