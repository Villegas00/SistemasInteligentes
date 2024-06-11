from turtle import width
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


def plot_dataset(X,y,axes):
    plt.plot(X[:,0][y==0],X[:,1][y==0],"b+")
    plt.plot(X[:,0][y==1],X[:,1][y==1],"go")
    plt.axis(axes) 
    plt.grid(True,axis='both')
    plt.title(r"Boundaries",fontsize=20)
    plt.xlabel(r"$x_1$",fontsize=18)
    plt.ylabel(r"$x_2$",fontsize=18,rotation=0)

def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0],axes[1],100)
    x1s = np.linspace(axes[2],axes[3],100)
    x0,x1 = np.meshgrid(x0s,x1s)
    X = np.c_[x0.ravel(),x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    plt.contourf(x0,x1,y_pred,cmap=plt.cm.brg,alpha=0.1)


data = pd.read_csv('01-boundaries.csv',header=0,names=['0', '1', '2'])
X = data.drop(['2'],axis=1).values
y = data['2']

neural_network= MLPClassifier(hidden_layer_sizes=(10, 10, 10, 10), max_iter=1000) 
neural_network.fit(X,y) 

plot_predictions(neural_network,[-2.5,2.5,-2.5,2.5])
plot_dataset(X,y,[-2.5,2.5,-2.5,2.5])

plt.show()