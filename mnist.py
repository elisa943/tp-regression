###############################################################################
# MODULES
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression

###############################################################################
# LOAD MNIST
###############################################################################
# Download MNIST
mnist = fetch_openml(data_id=554, parser='auto')
# copy mnist.data (type is pandas DataFrame)
data = mnist.data
# array (70000,784) collecting all the 28x28 vectorized images
img = data.to_numpy()
# array (70000,) containing the label of each image
lb = np.array(mnist.target,dtype=int)
# Splitting the dataset into training and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    img, lb, 
    test_size=0.25, 
    random_state=0)
# Number of classes
k = len(np.unique(lb))
# Sample sizes and dimension
(n,p) = img.shape
n_train = y_train.size
n_test = y_test.size 

###############################################################################
# TRAINING AND TEST SETS
###############################################################################

def Reg_log(X,y,X_test):
  rlog = LogisticRegression(max_iter=1000)
  rlog.fit(X,y)
  Y_pred=rlog.predict(X_test)
  return rlog,Y_pred

# Normaliser les données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrainer le modèle de régression logistique
rlog, y_pred = Reg_log(X_train,y_train,X_test)

# Calculer la matrice de confusion
cm = confusion_matrix(y_test, y_pred)
for i in range(0,len(cm)):
  cm[i,:]=100*(cm[i,:]).astype(float)/(np.sum(cm[i,:])).astype(float)
###############################################################################
# RETRAINING THE MODEL (Régularisation ℓ¹)
###############################################################################
def Reg_log_l1(X,y,X_test):
  rlog = LogisticRegression(penalty="l1", max_iter=1000, tol=0.01, solver="saga")
  rlog.fit(X,y)
  Y_pred=rlog.predict(X_test)
  return rlog,Y_pred

rlog_l1, y_pred_l1 = Reg_log_l1(X_train,y_train,X_test)
cm_l1 = confusion_matrix(y_test, y_pred_l1)
###############################################################################
# DISPLAY A SAMPLE
###############################################################################
m=25
plt.figure(figsize=(10,10))
for i in np.arange(m):
  ex_plot = plt.subplot(int(np.sqrt(m)),int(np.sqrt(m)),i+1)
  plt.imshow(X_test[i].reshape((28,28)), cmap='gray')
  ex_plot.set_xticks(()); ex_plot.set_yticks(())
  plt.title("Label = %i" % y_pred[i])

# Afficher les matrices de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rlog.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice de confusion pour la régression logistique l2")
print(100*sum(y_pred!=y_test)/len(y_pred))
print(100*sum(y_pred_l1!=y_test)/len(y_pred_l1))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_l1, display_labels=rlog_l1.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice de confusion pour la régression logistique l1")

# Afficher les coefficients beta 
coef = rlog.coef_
plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(5, 2, i + 1)
    plt.imshow(coef[i].reshape(28, 28), cmap='RdBu')
    plt.title(f'Classe {i}')
    plt.colorbar()
    plt.axis('off')
plt.suptitle("Coefficients βˆ pour chaque classe (l2)")

coef_l1 = rlog_l1.coef_
plt.figure(figsize=(10, 10))
for i in range(10):
    plt.subplot(5, 2, i + 1)
    plt.imshow(coef_l1[i].reshape(28, 28), cmap='RdBu')
    plt.title(f'Classe {i}')
    plt.colorbar()
    plt.axis('off')
plt.suptitle("Coefficients βˆ pour chaque classe (l1)")

plt.show()