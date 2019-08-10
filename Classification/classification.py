# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 11:43:48 2019

@author: Prem
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(classifier, x, y,resolution=0.05):
    import numpy as np
    import matplotlib.pyplot as plt
    # only for two features possible
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max,resolution),np.arange(x2_min, x2_max,resolution))
    xx = np.append(xx1.flatten()[:,np.newaxis],xx2.flatten()[:,np.newaxis],axis=1)
    yy = classifier.predict(xx).reshape(xx1.shape)
    
    # plot decision boundary
    plt.contourf(xx1, xx2, yy, alpha=0.4)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    unique_ = np.unique(y)
    for i in range(len(unique_)):
        xclass = x[np.where(y==unique_[i])]
        label = 'class '+str(i)
        plt.scatter(xclass[:,0],xclass[:,1],label=label) 
    plt.legend()
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()