import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

### data1

x = np.array(data1[0, :])
y = np.array(data1[1, :])

X=[[1,x[i]] for i in range(len(x))]

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

X=[[1,x[i]] for i in range(len(x))]

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

X =[[1, x[i], x[i]**2] for i in range(len(x))]
coeff = Reg_lin(X, y)
fx = coeff[1]*x_data + coeff[2] * x_data**2

# Plot outputs
plt.scatter(x, y, color="black")
plt.plot(x_data, fx, 'r-')
plt.title("Espace de redescription de data2")
plt.legend(["Données", "Espace de redescription"])
plt.show()