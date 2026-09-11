import bitbirch.bitbirch as bb

import numpy as np
import time
import sys

n=int(sys.argv[1])
features=2048
threshold=0.42
bf=3

np.random.seed(80)   
data=np.random.randint(2, size=(n,features))  
# data=np.load("../smi/BRD4_train_total.npy")
# rng=np.random.default_rng(3) 
# random_indices=rng.choice(data.shape[0], size=n, replace=False)
# data=data[random_indices] 

bb.set_merge("radius")
model=bb.BitBirch(threshold=threshold,
                  branching_factor=bf)
model.fit(data)
print("\nnum of leaf centroids: ", len(model.get_centroids()))

#nested sets
indices=[]
def build_nested_sets(node, counter, parent_id):
    print("length",len(node.subclusters_))
    if node is None:
        return 
    if node.subclusters_[0].child_ is None:
        print("1")
        indices.append([counter[0], counter[0]+1])
        print(indices)
        #parent_id+=1
        counter[0]+=2
        return
    
    
    print("2")
    indices.append([counter[0]])
    print(indices, parent_id)
    parent_id=len(indices)
    counter[0]+=1

    for i in node.subclusters_:
        build_nested_sets(i.child_, counter, parent_id)
    parent_id-=1
    print("3", parent_id)
    indices[parent_id].append(counter[0])
    counter[0]+=1
    print(indices)

build_nested_sets(model.root_, [1], 0)
print(indices)
print()

def recursive_travel(node):
    if node is None:
        return
    print(len(node.subclusters_))
    for i in node.subclusters_:
        print(i.mol_indices)
        recursive_travel(i.child_)

recursive_travel(model.root_)