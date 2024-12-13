import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression

###############################################################################
# Imports des données
###############################################################################
data1 = np.load('data1.npy')
data2 = np.load('data2.npy')
data3 = np.load('data3.npy')

len_data1 = len(data1[0, :])
len_data2 = len(data2[0, :])
len_data3 = len(data3[0, :])
###############################################################################


###############################################################################
# OLS
###############################################################################
def Reg_lin(X,y): # Régression Linéaire 
    coeff=np.linalg.inv(np.dot(np.transpose(X),X))
    coeff=np.dot(coeff,np.transpose(X))
    coeff=np.dot(coeff,y)
    return coeff

def l_OLS(u, v):
    return (u - v)**2

def erreur_apprentissage(y, y_pred,len_data):
    err = 1 / len_data * sum(l_OLS(y, y_pred))
    print("L'erreur d'apprentissage est de : ", err)
    
def X_create(q,x):
    X=[[x[i]**j for j in range(0,q+1)] for i in range(len(x))]
    return X

def OLS(x, y, len_data): 
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

### data1
x = np.array(data1[0, :])
y = np.array(data1[1, :])

x_test, y_test, y_pred, a = OLS(x, y, len_data1)

# Plot outputs
plt.figure()
plt.scatter(x, y, color="black")
plt.plot(x_test, y_pred, color="blue", linewidth=3)
plt.plot(x_test,a[1]*x_test+a[0], color="red")
plt.legend(["Données", "OLS avec formule","OLS"])
plt.title("Régression linéaire de data1")
plt.show()

erreur_apprentissage(y_test, y_pred, len_data1)

### data2

x = np.array(data2[0, :])
y = np.array(data2[1, :])
q=1
#X=[[1,x[i]] for i in range(len(x))]
X=X_create(q,x)
x_data = x
y_data = y

# Use only one feature
x_data = x_data[:, np.newaxis]

# Split the data into training/testing sets
x_train = x_data[:-len_data2//2]
x_test = x_data[-len_data2//2:]

# Split the targets into training/testing sets
y_train = y_data[:-len_data2//2]
y_test = y_data[-len_data2//2:]

# Create linear regression object
regr = LinearRegression()
a=Reg_lin(X,y)

# Train the model using the training sets
regr.fit(x_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(x_test)

# Plot outputs
plt.scatter(x, y, color="black")
plt.plot(x_test, y_pred, color="blue", linewidth=3)
plt.plot(x_test,a[1]*x_test+a[0], color="red")
plt.legend(["Données", "OLS avec formule","OLS"])
plt.title("Régression linéaire de data2")
plt.show()

erreur_apprentissage(y_test, y_pred, len_data2)

###############################################################################
# Espace de redescription
###############################################################################

### data2
x = np.array(data2[0, :])
y = np.array(data2[1, :])

x_data = np.sort(x)
y_data = y

# Split the data into training/testing sets
x_train = x_data[:-len_data2//2]
x_test = x_data[-len_data2//2:]

# Split the targets into training/testing sets
y_train = y_data[:-len_data2//2]
y_test = y_data[-len_data2//2:]
q=2
#X =[[1, x[i], x[i]**2] for i in range(len(x))]
X=X_create(q,x)
coeff = Reg_lin(X, y)
fx = coeff[1]*x_data + coeff[2] * x_data**2

# Plot outputs
plt.scatter(x, y, color="black")
plt.plot(x_data, fx, 'r-')
plt.title("Espace de redescription pour data2")
plt.legend(["Données", "Espace de redescription"])
plt.show()

### data 3

# Chargement data 3
x = np.array(data3[0, :])
y = np.array(data3[1, :])

x_data = np.sort(x)
y_data = y
q=10
# Split the data into training/testing sets
x_train = x_data[:-len_data2//2]
x_test = x_data[-len_data2//2:]

# Split the targets into training/testing sets
y_train = y_data[:-len_data2//2]
y_test = y_data[-len_data2//2:]

#X =[[1, x[i], x[i]**2, x[i]**3, x[i]**4, x[i]**5, x[i]**6, x[i]**7, x[i]**8, x[i]**9, x[i]**10] for i in range(len(x))]
X=X_create(q,x)
coeff = Reg_lin(X, y)
fx=coeff[0]
for i in range(1,q+1):
    fx = fx + coeff[i]*(x_data**i)


plt.scatter(x, y, color="black")
plt.plot(x_data, fx, 'r-')
plt.title("data3")
plt.legend(["Données", "Espace de redescription"])
plt.show()

###############################################################################
# Methode LASSO
###############################################################################

# Chargement data 3
x = np.array(data3[0, :])
y = np.array(data3[1, :])

x2=np.sort(x)

xdata=x
ydata=y
q=1
X=X_create(q,x)
coeff = Reg_lin(X, y)
fx=coeff[0]
for i in range(1,q+1):
    fx = fx + coeff[i]*(x2**i)  
xdata = np.vstack([np.ones(len(xdata)), xdata]).T
x_train = xdata[:-len_data3//2]
x_test = xdata[-len_data3//2:]
y_train = ydata[:-len_data3//2]
clf=Lasso(alpha=0.1)
clf.fit(x_train,y_train)
print(clf.intercept_,clf.coef_)

plt.scatter(x, y, color="black")
plt.plot(x2, fx, 'r-')
plt.plot(x2, clf.intercept_ + x2 * clf.coef_[1], color="blue", linewidth=3)
plt.legend(["Données", "OLS","LASSO"])
plt.title("Régression linéaire de data3")
plt.show()

###############################################################################
# Ridge 
###############################################################################

from sklearn.linear_model import Ridge

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

x = np.array(data3[0, :])
y = np.array(data3[1, :])
lamb = 2.0

intercept, coef = RIDGE(x, y, lamb)
x_test, y_test, y_pred, a = OLS(x, y, len_data3)

# Afficher les points de données et la ligne de régression
plt.figure()
plt.scatter(x, y, color='black')
plt.plot(x, intercept + coef[1] * x, color='red', label='Ridge regression line')
plt.plot(x_test, y_pred, color="green", label="OLS de Python")
plt.plot(x_data,a[1]*x_data+a[0], color="blue", label="OLS")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Approches OLS et Ridge')
plt.show()

erreur_apprentissage(y_test, y_pred, len_data3)

