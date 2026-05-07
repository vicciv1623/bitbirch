import numpy as np
import sys
from pathlib import Path

sklearn_path=Path('/home/honestkids/Documents/MQ_lab_research/scikit-learn')
sys.path.insert(0,str(sklearn_path))
from sklearn.cluster import KMeans

n=10
features=50
np.random.seed(67)
#data=np.random.randint(2, size=(n,features))
data=np.array([[1,1,0,1,1,1,0,0,0,0],
              [1,1,1,1,1,0,0,1,0,1],
              [1,0,0,1,1,0,0,0,0,0],
              [1,1,1,1,1,0,0,1,0,0],
              [1,1,1,1,1,0,0,0,0,0],
              [0,0,0,0,0,1,1,1,1,1],
              [0,0,0,0,0,1,1,1,1,1],
              [0,0,0,0,0,1,1,1,1,1],
              [0,0,0,0,0,1,1,1,1,1],
              [0,0,0,0,0,1,1,1,1,1]])
rng=np.random.default_rng(67)
#data=rng.integers(2, size=(n,features))

model=KMeans(n_clusters=2, init="random")
labels=model.fit_predict(data)
print(labels)