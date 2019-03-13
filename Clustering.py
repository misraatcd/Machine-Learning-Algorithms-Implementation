import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
#from scipy.spatial import distance

def distance(X,mu):
  # calculate the euclidean distance between numpy arrays X and mu
  #print(X.shape)
  #print(mu.shape)
  #(k,n)=mu.shape # k is number of centres
  #(k,n)=X.shape # m is number of data points
  
  z = np.array((X - mu))
  print(z)
  z1 = np.square(z)
  print(z1)
  d1 = np.sum(z1, axis = 1)
  print(d1)
  
  ##### insert your code here #####
  return d1
	
def findClosestCentres(X,mu):
  # finds the centre in mu closest to each point in X
  (k,n)=mu.shape # k is number of centres
  (m,n)=X.shape # m is number of data points
  #print(k)
  #print(m)
  #C=list()
  #j = 1
  #for j in range(list()):
  #  d = distance(mu, X[j])
  #  C.append((X[j])) 
    
  #C.sort(key=operator.itemgetter(1)) 
  
  #C = list()
  #for point in X:
  #    distances = list()
  #    for centroid in mu:
  #        distances.append(sp.sum((point[0] - centroid[0])**2 + (point[1] - centroid[1])**2))
        
  #    C.append(sp.array(distances).argmin())
  #return C
  #N=[]
  #for x in range(k):
  #  N.append(C[x][0])
  ##### insert your code here #####
  #return C
  C=list()
  Dis=list()
  
  for a in range(k):
    Dis.append(distance(X,np.tile(mu[a,],[np.size(X,0),1])))
  
  Dis=np.argmin(Dis,axis=0)

  for b in range(k):
    C.append(np.squeeze(np.argwhere(Dis==b), axis=1).tolist())

  return C
    
def updateCentres(X,C):
  # updates the centres to be the average of the points closest to it.  
  k=len(C) # k is number of centres
  (m,n)=X.shape # n is number of features
  ##### insert your code here #####
  mu =list()
  for i in range(k):
    mu.append(np.mean(np.take(X,C[i],axis=0),axis=0))
  mu =np.array(mu)
  return mu

def plotData(X,C,mu):
  # plot the data, coloured according to which centre is closest. and also plot the centres themselves
  fig, ax = plt.subplots(figsize=(12,8))
  ax.scatter(X[C[0],0], X[C[0],1], c='c', marker='o')
  ax.scatter(X[C[1],0], X[C[1],1], c='b', marker='o')
  ax.scatter(X[C[2],0], X[C[2],1], c='g', marker='o')
  # plot centres
  ax.scatter(mu[:,0], mu[:,1], c='r', marker='x', s=100,label='centres')
  ax.set_xlabel('x1')
  ax.set_ylabel('x2')  
  ax.legend()
  fig.savefig('graph.png') 
  
def main():
  print('testing the distance function ...')
  print(distance(np.array([[1,2],[3,4]]), np.array([[1,2],[1,2]])))
  print('expected output is [0,8]')
  
  print('testing the findClosestCentres function ...')
  print(findClosestCentres(np.array([[1,2],[3,4],[0.9,1.8]]),np.array([[1,2],[2.5,3.5]])))
  print('expected output is [[0,2],[1]]')

  print('testing the updateCentres function ...')
  print(updateCentres(np.array([[1,2],[3,4],[0.9,1.8]]),[[0,2],[1]]))
  print('expected output is [[0.95,1.9],[3,4]]')

  print('loading test data ...')
  X=np.loadtxt('data.txt')
  [m,n]=X.shape
  iters=10
  k=3
  print('initialising centres ...')
  init_points = np.random.choice(m, k, replace=False)
  mu=X[init_points,:] # initialise centres randomly
  print('running k-means algorithm ...')
  for i in range(iters):
    C=findClosestCentres(X,mu)
    mu=updateCentres(X,C)
  print('plotting output')
  plotData(X,C,mu)  
  
if __name__ == '__main__':
  main()