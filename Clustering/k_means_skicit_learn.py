from turtle import width
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans

def plot_dataset(X,y,axes):
    plt.plot(X[:,0][y=="Head"],X[:,1][y=="Head"],"b+")
    plt.plot(X[:,0][y=="Ear_right"],X[:,1][y=="Ear_right"],"go")
    plt.plot(X[:,0][y=="Ear_left"],X[:,1][y=="Ear_left"],"kv")
    plt.axis(axes) 
    plt.grid(True,axis='both')
    plt.title(r"Mouse",fontsize=20)
    plt.xlabel(r"$x_1$",fontsize=18)
    plt.ylabel(r"$x_2$",fontsize=18,rotation=0)

def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0],axes[1],100)
    x1s = np.linspace(axes[2],axes[3],100)
    x0,x1 = np.meshgrid(x0s,x1s)
    X = np.c_[x0.ravel(),x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    plt.contourf(x0,x1,y_pred,cmap=plt.cm.brg,alpha=0.1)


data = pd.read_csv('01-mouse.csv',header=None,comment="#",sep=' ')
X = data.drop([2],axis=1).values
y = data[2]

k_means=KMeans(n_clusters=3)
k_means.fit(X,y)

plot_predictions(k_means,[0,1,0,1])
plot_dataset(X,y,[0,1,0,1])

plt.show()