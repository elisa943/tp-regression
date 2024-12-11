import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso


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

### data1

x = np.array(data1[0, :])
y = np.array(data1[1, :])
q=1
#X=[[1,x[i]] for i in range(len(x))]
X=X_create(q,x)
x_data = x
y_data = y

# Use only one feature
x_data = x_data[:, np.newaxis]

# Split the data into training/testing sets
x_train = x_data[:-len_data1//2]
x_test = x_data[-len_data1//2:]

# Split the targets into training/testing sets
y_train = y_data[:-len_data1//2]
y_test = y_data[-len_data1//2:]

# Create linear regression object
regr = LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)
a=Reg_lin(X,y)

# Make predictions using the testing set
y_pred = regr.predict(x_test)

# Plot outputs
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
q=1
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
    
x_data = x_data[:, np.newaxis]
x_train = x_data[:-len_data3//2]
x_test = x_data[-len_data3//2:]
y_train = y_data[:-len_data3//2]

clf=Lasso()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)

plt.scatter(x, y, color="black")
plt.plot(x2, fx, 'r-')
plt.plot(x_test, y_pred, color="blue", linewidth=3)
plt.legend(["Données", "OLS","LASSO"])
plt.title("Régression linéaire de data3")
plt.show()