
import pandas as pd
import numpy as np
from itertools import cycle,islice
import matplotlib.pyplot as plt

#matplotlib.get_backend()

d={'x1':pd.Series(np.array([1,2,4,5]),['a','b','c','d']), 'x2': pd.Series(np.array([1,1,3,4]),['a','b','c','d'])}
data=pd.DataFrame(d)
plt.figure()
#print(data)
#plt.plot(x=data['x1'],y=data['x2'])
plt.scatter(d['x1'], d['x2'])
plt.show()
#dist=metric.DistanceMetric("euclidean")
#dist.pairwise(d['x1'],d['x2'])
L1=[1,1]
L2=[5,4]
data['centroid1']=np.linalg.norm(data[['x1', 'x2']].sub(np.array(L1)), axis=1)
data['centroid']=np.linalg.norm(data[['x1', 'x2']].sub(np.array(L2)), axis=1)

#data['attributes']=(data['centroid1'],data['centroid'])
print(data)






