import bitbirch.bitbirch as bb
import bitbirch.bitbirch_partition as bb_part
import numpy as np
import time
import sys

n=int(sys.argv[1])
features=2048
threshold=0.4
bf=50

np.random.seed(69)   
data=np.random.randint(2, size=(n,features))   

bb.set_merge("radius")
model=bb.BitBirch(threshold=threshold,
                  branching_factor=bf)
start=time.perf_counter()
model.fit(data)
end=time.perf_counter()
print("num of leafs: ", len(model.get_centroids()))
print("time: ", end-start)

bb_part.set_merge("radius")
model_p=bb_part.BitBirch(threshold=threshold,
                         branching_factor=bf,
                         tolerance=0.01)
start=time.perf_counter()
model_p.fit(data)
end=time.perf_counter()
print("\nnum of leafs: ", len(model_p.get_centroids()))
list_mol_ids=model_p.get_cluster_mol_ids()
leaf_mol=0
for i in list_mol_ids:
    leaf_mol+=len(i)
print("noise: ", len(model_p.noise))
print("non noise: ", leaf_mol)
print(leaf_mol + len(model_p.noise))
print("time: ", end-start)