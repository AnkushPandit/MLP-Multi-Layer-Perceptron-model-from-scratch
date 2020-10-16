import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
from sklearn.utils import shuffle

ds1=pd.read_csv("C:\\Users\\Ankush\\Desktop\\ML Assignment\\Assignment 1\\input.csv") #path in local device
ds=shuffle(ds1)
X=ds.iloc[:,0:3].values
Y=ds.iloc[:,3:5].values

#plotting data points
for i in range(len(X)):
    if Y[i][1] ==1:
        plt.scatter(X[i][1],X[i][2],color='r',marker='*')
    else:
        plt.scatter(X[i][1],X[i][2],color='g')

#if you uncomment the below line you will get the figure same as input_plot.jpg in main repository
#plt.show()

lr=0.1 #learning rate
N=len(ds) #no. of data points
H=4 #no. of nodes in hidden layer
x=3 #no. of input parameters: x0,x1,x2
op=2 #no. of nodes in output layer

u1=np.zeros((N,H+1))
v1=np.zeros((N,H+1))

u2=np.zeros((N,op))
v2=np.zeros((N,op))

w1=np.random.random((H+1)*x)
w1.shape=(H+1,x)
w2=np.random.random(op*(H+1))
w2.shape=(op,H+1)

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

#training the model
epoch=1000
for e in range(epoch):
    er=0
    for n in range(N-10): #10 data points are left for testing
        en=0
        
        #v1[n][0]=1
        for j in range(1,H+1):
            u1[n][j]=0
            for i in range(x):
                u1[n][j]+=X[n][i]*w1[j][i]
                v1[n][j]=sigmoid(u1[n][j])

        v1[n][0]=1
        
        for j in range(op):
            u2[n][j]=0
            for i in range(H+1):
                u2[n][j]+=v1[n][i]*w2[j][i]
                v2[n][j]=sigmoid(u2[n][j])
                en+=(Y[n][j]-v2[n][j])*(Y[n][j]-v2[n][j])*0.5

        er+=en

        for k in range(op):
            for j in range(H+1):
                w2[k][j]+=lr*(Y[n][k]-v2[n][k])*v2[n][k]*(1-v2[n][k])*v1[n][j]

        for j in range(H+1):
            for i in range(x):
                temp=0
                for k in range(op):
                    temp+=(Y[n][k]-v2[n][k])*v2[n][k]*(1-v2[n][k])*w2[k][j]
                w1[j][i]+=lr*temp*v1[n][j]*(1-v1[n][j])*X[n][i]

    #print(er/N)

err=0

# the testing is done on whole dataset but you can use the only 10 points left for this purpose
for n in range(N):
    for j in range(1,H+1):
        u1[n][j]=0
        for i in range(x):
            u1[n][j]+=X[n][i]*w1[j][i]
            v1[n][j]=sigmoid(u1[n][j])


    v1[n][0]=1
    for j in range(op):
        u2[n][j]=0
        for i in range(H+1):
            u2[n][j]+=v1[n][i]*w2[j][i]
            v2[n][j]=sigmoid(u2[n][j])

    #err+=(v2[n][0]-Y[n][0])*(v2[n][0]-Y[n][0])+(v2[n][1]-Y[n][1])*(v2[n][0]-Y[n][1])

    print(v2[n],Y[n])


