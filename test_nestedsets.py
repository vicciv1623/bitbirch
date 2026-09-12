import bitbirch.bitbirch as bb

import numpy as np
import time
import sys
import pickle
import base64
import zlib
import struct
from bisect import bisect_left
from collections.abc import Iterable
from collections import namedtuple

n=int(sys.argv[1])
features=2048
threshold=0.42
bf=50

#np.random.seed(80)   
#data=np.random.randint(2, size=(n,features))  
data=np.load("../smi/BRD4_train_total.npy")
rng=np.random.default_rng(4) 
random_indices=rng.choice(data.shape[0], size=n, replace=False)
data=data[random_indices] 
print(data)

bb.set_merge("radius")
model=bb.BitBirch(threshold=threshold,
                  branching_factor=bf)
start=time.perf_counter()
model.fit(data)
end=time.perf_counter()
print("\nnum of leaf centroids: ", len(model.get_centroids()))
print("time: ", end-start)

#sanity checking functions
target=[]
def recursive_travel(node, target):
    if node is None:
        return
    for i in node.subclusters_:
        if len(i.mol_indices)==61:
            target.append(i.centroid_)
        recursive_travel(i.child_, target)

recursive_travel(model.root_, target)

def get_pkl_memry(obj):
    pickled_bytes = pickle.dumps(obj)
    byte_size = len(pickled_bytes)
    print(f"Pickled size: {byte_size / 1024:.2f} KB")



#compression function
class CappedIntBlob(list[int]):
    #https://www.janmeppe.com/blog/how-to-compress-a-list-of-integers-in-python/
    """Capped integer Base64 Large Object aka CappedIntBlob.
    
    Represents a list of 2-byte unsigned short integers, capped at max_val

    When converted to str, will convert all integers into 2-byte values and then base64 encode them after compressing with zlib.

    The constructor accepts either a base64 string or an iterable of integers.
    """

    def __init__(self, contents: str | Iterable[int], max_val: int = 1e9):
        """Constructor
        @param contents: the contents, either a list of ints or a base64 string with 2-byte unsigned short integer representation.
        """
        self.max = max_val
        self.lst: Iterable[int]
        if isinstance(contents, str):
            # If input is a string, decode and decompress it
            bs: bytes = zlib.decompress(base64.b64decode(contents))
            # < = little endian, H = unsigned short with 2-byte size integers
            self.lst = struct.unpack("<" + "H" * int(len(bs) / 2), bs)
        elif isinstance(contents, Iterable):
            self.lst = [min(int(x), self.max) for x in contents]
        else:
            raise ValueError("Expecting a string or an iterable for contents")

        super().__init__(self.lst)

    def __str__(self):
        """Convert to string
        @return: base64 string encoding for list of ints in 2 byte unsigned short integer representation.
        """
        return base64.b64encode(zlib.compress(struct.pack("<" + "H" * len(self), *self))).decode("utf-8")


def bisect_left_search(lst, elem):
    i = bisect_left(lst, elem)
    if i != len(lst) and lst[i] == elem:
        return i
    return -1


#nested sets
def retrieve_root(model):
    '''
    This function retrieves and stores the centroids of the root nodes 
    in BitBIRCH for easy access in the future.
    
    Input: BitBIRCH model (after fitted with data)
    Output: np.ndarray where each row corresponds to each root subcluster; 
            assumed to have 2048 features 
    '''
    root_node=np.empty((len(model.root_.subclusters_), 2048), dtype=np.uint64)
    for ind, i in enumerate(model.root_.subclusters_):
        root_node[ind]=i.centroid_

    return root_node

NS_stats=namedtuple("NS_stats", ["indices", "leaf_stats", "leaf_idx", "root_stats"])

def build_nested_sets(model):
    '''
    Docstring for build_nested_sets

    param model: BitBIRCH model (after fitted with data)

    Output ns_stats: a namedtuple storing three pieces of information: indices, leaf_stats, leaf_idx, root_stats  
                        - indices: check above Docstring for more information
                        - leaf_stats: check above Docstring for more information
                        - leaf_idx: check above Docstring for more information
                        - root_stats: np.ndarray where each row corresponds to each root subcluster; 
                                      assumed to have 2048 features 
    '''

    def build_nested_sets_helper(node, counter, parent_id, indices, leaf_stats, leaf_idx):
        '''
        Docstring for build_nested_sets_helper
        This helper function generates the nested sets indices of BitBIRCH
        
        :param node: Node object
        :param counter: list[int] of size 1; will be responsible for keeping track  
        :param parent_id: int; will ALWAYS be initiated with value of 0
        :param indices: list[list[int]]; stores the finalized indicecs for successful nested sets
        :param leaf_stats: list[np.ndarray, list[list[int]]]: np.ndarray stores the centroids of the leaf node 
                            and list[list[int]] stores the mol_indices of each subcluster within leaf node
        :param leaf_idx: list[int]; stores the left indices pertaining to leaf nodes  
        '''
        if node is None:
            return 
        if node.subclusters_[0].child_ is None:  #this is a leaf node
            indices.append((counter[0], counter[0]+1))
            leaf_idx.append(counter[0])
            counter[0]+=2

            lin_sums=np.empty((len(node.subclusters_), 2048), dtype=np.uint64)
            mol_inds=[]
            for ind, sub in enumerate(node.subclusters_):
                lin_sums[ind]=sub.linear_sum_
                mol_inds.append(sub.mol_indices)
            leaf_stats.append([lin_sums, mol_inds])
            return
        
        indices.append([counter[0]])
        parent_id=len(indices)
        counter[0]+=1

        for i in node.subclusters_:
            build_nested_sets_helper(i.child_, counter, parent_id, indices, leaf_stats, leaf_idx)

        parent_id-=1
        indices[parent_id].append(counter[0])
        indices[parent_id]=tuple(indices[parent_id])
        counter[0]+=1


    indices=[]
    counter=[0]
    parent_id=0

    leaf_stats=[]
    leaf_idx=[]
    root_stats=retrieve_root(model)

    build_nested_sets_helper(model.root_, counter, parent_id, indices, leaf_stats, leaf_idx)

    ns_stats=NS_stats(indices, leaf_stats, leaf_idx, root_stats)
    return ns_stats

##finding higher node details ground up:
def ground_up(left_id, right_id, ns_stats):
    '''
    Docstring for ground_up
    This function returns the centroids of a node identified by the left and right indices

    :param left_id: the left index of the node
    :param right_id: the right index of the node
    :param ns_stats: namedtuple output from build_nested_sets function

    Output np.ndarray of centroids pertaining to the node
    '''

    def ground_up_helper(left_id, right_id, ns_stats):
        '''
        Docstring for ground_up_helper
    
        :param left_id: the left index of child node of subcluster within given node
        :param right_id: the left index of child node of subcluster within given node
        :param ns_stats: check above Docstring for more information

        Output np.array of a centroid of a subcluster
        '''
        tot_lin_sum=np.empty(2048, dtype=np.uint64)
        tot_num_mol=0

        for left, right in ns_stats.indices:
            if left>left_id and right<right_id:
                if right-left==1:   #is a leaf
                    index=bisect_left(ns_stats.leaf_idx, left)

                    # add stuff
                    tot_lin_sum+=np.sum(ns_stats.leaf_stats[index][0], axis=0)
                    tot_num_mol+=np.sum([len(x) for x in ns_stats.leaf_stats[index][1]])
        
        #print(tot_num_mol)
        return bb.calc_centroid(tot_lin_sum, tot_num_mol)

    centroids=[]

    start_left=left_id+1
    left_indices=[x[0] for x in ns_stats.indices]
    
    while start_left<right_id:

        index=bisect_left(left_indices, start_left)
        start_right=ns_stats.indices[index][1]

        centroid=ground_up_helper(start_left, start_right, ns_stats)
        centroids.append(centroid)

        start_left=start_right+1
    
    return np.array(centroids)


#inserting a molecule (without modifying the tree)
def insert_static(new_mol, ns_stats):
    a = np.dot(ns_stats.root_stats, new_mol)
    sim_matrix = a / (np.sum(ns_stats.root_stats, axis = 1) + np.sum(new_mol) - a)
    closest_index = np.argmax(sim_matrix)

    #find corresponding indices: starting at root level
    left_indices=[x[0] for x in ns_stats.indices]
    start_left=1
    for i in range(closest_index):
        index=bisect_left(left_indices, start_left)
        start_left=ns_stats.indices[index][1]+1

    index=bisect_left(left_indices, start_left)
    start_right=ns_stats.indices[index][1]

    while start_right-start_left!=1:
        centroids=ground_up(start_left, start_right, ns_stats)

        a = np.dot(centroids, new_mol)
        sim_matrix = a / (np.sum(centroids, axis = 1) + np.sum(new_mol) - a)
        closest_index = np.argmax(sim_matrix)

        for _ in range(closest_index):
            index=bisect_left(left_indices, start_left+1)
            start_left=ns_stats.indices[index][1]+1
            index=bisect_left(left_indices, start_left)
            start_right=ns_stats.indices[index][1]

        start_left=start_left+1
        index=bisect_left(left_indices, start_left)
        start_right=ns_stats.indices[index][1]

    index=bisect_left(ns_stats.leaf_idx, start_left)
    return ns_stats.leaf_stats[index]



start=time.perf_counter()
ns_stats=build_nested_sets(model)
end=time.perf_counter()
print("time", end-start)

start=time.perf_counter()
final_leaf=insert_static(model.root_.subclusters_[0].centroid_, ns_stats)
end=time.perf_counter()
print("time", end-start)
print(final_leaf)


get_pkl_memry(model)
get_pkl_memry(ns_stats)

