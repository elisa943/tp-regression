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
  rlog = LogisticRegression()
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

###############################################################################
# DISPLAY A SAMPLE
###############################################################################
m=16
plt.figure(figsize=(10,10))
for i in np.arange(m):
  ex_plot = plt.subplot(int(np.sqrt(m)),int(np.sqrt(m)),i+1)
  plt.imshow(img[i,:].reshape((28,28)), cmap='gray')
  ex_plot.set_xticks(()); ex_plot.set_yticks(())
  plt.title("Label = %i" % lb[i])

# Afficher la matrice de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rlog.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrice de confusion pour la régression logistique")

plt.show()