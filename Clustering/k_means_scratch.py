from turtle import distance, width
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def init_centroids(X,n_clusters):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    centroids = np.zeros((n_clusters, n_features), dtype=np.float64)
    np.random.seed(1000000)
    for i in range (0,n_clusters):
        index = np.random.randint(0, n_samples)
        centroids[i,:] = X[index,:]
    return centroids

def compute_distance(X, centroids, n_clusters):
    distance = np.zeros((X.shape[0], n_clusters))
    for k in range(n_clusters):
        distance [:,k] = np.square(np.linalg.norm(X-centroids[k,:], axis=1))
    return distance

def find_closest_cluster(distance):
    return np.argmin(distance, axis=1)

def update_centroids(X, labels, n_clusters):
    centroids = np.zeros((n_clusters, X.shape[1]))
    for k in range(n_clusters):
        centroids[k,:] = np.mean(X[labels==k, :], axis=0)
    return centroids

