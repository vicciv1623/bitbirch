import bitbirch.bitbirch as bb_
import bitbirch.bb_entire_old as bb_
import bitbirch.bitbirch_k_copy as bb

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
bf=3

#np.random.seed(80)   
#data=np.random.randint(2, size=(n,features))  
data=np.load("../smi/BRD4_train_total.npy")
rng=np.random.default_rng(3) 
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

def bisect_left_search(lst, elem):
    i = bisect_left(lst, elem)
    if i != len(lst) and lst[i] == elem:
        return i
    return -1



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
            self.lst = struct.unpack("<" + "q" * int(len(bs) / 8), bs)
        elif isinstance(contents, Iterable):
            self.lst = [min(int(x), self.max) for x in contents]
        else:
            raise ValueError("Expecting a string or an iterable for contents")

        super().__init__(self.lst)

    def __str__(self):
        """Convert to string
        @return: base64 string encoding for list of ints in 2 byte unsigned short integer representation.
        """
        return base64.b64encode(zlib.compress(struct.pack("<" + "q" * len(self), *self))).decode("utf-8")


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

NS_stats=namedtuple("NS_stats", ["indices", "leaf_stats", "leaf_idx", "root_stats", "reordered_indices"])

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
                mol_inds.append(str(CappedIntBlob(sub.mol_indices)))
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

    ns_stats=NS_stats(indices, leaf_stats, leaf_idx, root_stats, 0)
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

#reordering ns_stats.indices into more readable format
def reorder_ns_stats_indices_(ns_stats):
    '''
    Docstring for reorder_ns_stats_indices
    This function essentially reorders the ns_stats.indices into a more understandible
    format. It will categorize the indices by its correspondng the levels. For example,
    the root node will be the first element, then all the nodes pertaining in the level beneath, 
    then so on until the leaf nodes which constitude at the end of the list
    
    :param ns_stats: Description

    Output: a new NS_stats tuple
    '''
    og_indices=ns_stats.indices
    left_indices=[left for left,right in og_indices]   #this is going to be in order so we can use binary
    reordered=[og_indices[0]]

    if len(og_indices)==1:
        print("There is only one node anyways so no points in reordering.")
        return
    
    anchor_left=og_indices[1][0]
    anchor_right=og_indices[1][1]

    while anchor_right - anchor_left >1:
        num_of_nodes=1
        reordered.append((anchor_left, anchor_right))

        target=anchor_right+1

        #grouping and reordering the nodes by level 
        while target < og_indices[0][1]: # max value
            next_left=bisect_left_search(left_indices, target)
            reordered.append(og_indices[next_left])
            num_of_nodes+=1

            target=og_indices[next_left][1] +1

        anchor_left+=1
        print("target", target)
        anchor_left_idx=bisect_left_search(left_indices, anchor_left)

        if anchor_left_idx<0:
            break

        anchor_right=og_indices[anchor_left_idx][1]
        print("anchor rihgt", anchor_right)

    #add the remaining leaf nodes indices
    for i in ns_stats.leaf_idx:
        reordered.append((i, i+1))
    
    new_ns_stats=NS_stats(ns_stats.indices, ns_stats.leaf_stats, ns_stats.leaf_idx, 
                          ns_stats.root_stats, reordered)
    return new_ns_stats


def reorder_ns_stats_indices(ns_stats):
    og_indices=ns_stats.indices
    left_indices=[left for left,right in og_indices]   #this is going to be in order so we can use binary
    reordered=[og_indices[0]]   #add root node as first element

    if len(og_indices)==1:
        print("There is only one node anyways so no point in reordering.")
        return
    
    anchor_left=og_indices[1][0]
    anchor_right=og_indices[1][1]

    num_nodes=1    #this variable stores how many nodes are in nth level
    start=0        #this variable is to indicate starting position 

    while anchor_right-anchor_left > 1:
        next_num_nodes=0
        for left, right in reordered[start:start+num_nodes]:
            target=left+1

            while target < right:
                next_left=bisect_left_search(left_indices, target)
                reordered.append(og_indices[next_left])
                
                next_num_nodes+=1
                target=og_indices[next_left][1]+1

        start+=num_nodes
        num_nodes=next_num_nodes

        anchor_left=reordered[-1][0]
        anchor_right=reordered[-1][1]

    new_ns_stats=NS_stats(ns_stats.indices, ns_stats.leaf_stats, ns_stats.leaf_idx, 
                          ns_stats.root_stats, reordered)
    return new_ns_stats


def print_ns_stats_indices(ns_stats):
    '''
    Docstring for print_ns_stats_indices
    This function prints out the indices of the non-leafnodes for better understanding of the 
    hierarchical structure. Ideally use this function only after invoking reorder_ns_stats_indices 
    
    :param ns_stats: Description

    Output: void
    '''
    right_max=ns_stats.reordered_indices[0][1]
    level=0
    if level==0:
        print("Root node")
    for left, right in ns_stats.reordered_indices:
        if right-left==1:
            print("Leaves")
            break

        print(f"{left:<5}  {right}")

        if right==right_max:
            print("-------------")
            right_max-=1
            level+=1

            print(f"Level {level}")

            
#inserting a molecule (without modifying the tree)
def insert_static(new_mol, ns_stats):
    '''
    Docstring for insert_static
    This function inserts a new molecule into the tree and returns the 
    centroid and mol_indices pertaining to the leaf cluster the molecule 
    would have landed. It does NOT modify or save the new molecule into the tree.
    
    :param new_mol: np.array; of size 2048 features, describing the SMILES 
                    fingerprint of the molecule you want to insert to tree
    :param ns_stats: namedtuple output from build_nested_sets function

    Output: np.ndarray, list[list[int]]: the first element is the linear sum and the
                second element is molecular indices of the final leaf subcluster 
    '''

    #enter at the root node
    a = np.dot(ns_stats.root_stats, new_mol)
    sim_matrix = a / (np.sum(ns_stats.root_stats, axis = 1) + np.sum(new_mol) - a)
    closest_index = np.argmax(sim_matrix)

    if len(ns_stats.indices)==1:   #there is actually one leaf node for this tree
        print(closest_index)
        return ns_stats.leaf_stats[0][0][closest_index], ns_stats.leaf_stats[0][1][closest_index]

    #find corresponding indices: starting at root level
    left_indices=[x[0] for x in ns_stats.indices]
    start_left=1
    for i in range(closest_index):
        index=bisect_left(left_indices, start_left)
        start_left=ns_stats.indices[index][1]+1

    index=bisect_left(left_indices, start_left)
    start_right=ns_stats.indices[index][1]

    #find the child node to enter through
    #traverse down the tree
    while start_right-start_left!=1:
        centroids=ground_up(start_left, start_right, ns_stats)

        a = np.dot(centroids, new_mol)
        sim_matrix = a / (np.sum(centroids, axis = 1) + np.sum(new_mol) - a)
        closest_index = np.argmax(sim_matrix)

        #keep going down child nodes
        for _ in range(closest_index):
            index=bisect_left(left_indices, start_left+1)
            start_left=ns_stats.indices[index][1]+1

            index=bisect_left(left_indices, start_left)
            start_right=ns_stats.indices[index][1]

        start_left=start_left+1
        index=bisect_left(left_indices, start_left)
        start_right=ns_stats.indices[index][1]

    #we've reached the leaf level, almost to there
    #identify the leaf node so we can do one more similarity
    index=bisect_left(ns_stats.leaf_idx, start_left)

    a = np.dot(ns_stats.leaf_stats[index][0], new_mol)
    sim_matrix = a / (np.sum(ns_stats.leaf_stats[index][0], axis = 1) + np.sum(new_mol) - a)
    closest_index = np.argmax(sim_matrix)

    final_ls=ns_stats.leaf_stats[0][0][closest_index]
    final_mol_ind=CappedIntBlob(ns_stats.leaf_stats[0][1][closest_index]).lst
    return final_ls, final_mol_ind 

def retrieve_k_level_nodes(level_k, ns_stats):
    '''
    Docstring for retrieve_k_level_nodes
    This function returns the pertaining centroids of all the nodes at the level of interest

    :param k: int, the level of interest
    :param ns_stats: NS_Stats named tuple

    Output: list[np.ndarray]: of the centroids 
    '''

    #check if requested level is even possible
    leaf_level=ns_stats.leaf_idx[0]
    if level_k==leaf_level:
        print("The requested level is at the leaf level.")
    elif level_k>leaf_level:
        print(f"You have too few levels. Please input a number smaller or equal to {leaf_level}.")
        return
    
    indices=ns_stats.reordered_indices
    nodes=[]
    nodes_id=[]

    target_idx=0
    for i, index in enumerate(indices):
        if index[0]==level_k:
            target_idx=i
            break
    
    print(target_idx)
    #only need to check the contingous indices in that section 
    while indices[target_idx][0]!=level_k+1:
        nodes.append(ground_up(indices[target_idx][0],
                               indices[target_idx][1],
                               ns_stats))
        nodes_id.append(indices[target_idx][0])
        target_idx+=1
        print(target_idx)

    return nodes, nodes_id

def insert_static_at_k_level(new_mol, level_k, ns_stats):
    nodes, nodes_id=retrieve_k_level_nodes(level_k, ns_stats)
    concat_nodes=np.vstack(nodes)

    a = np.dot(concat_nodes, new_mol)
    sim_matrix = a / (np.sum(concat_nodes, axis = 1) + np.sum(new_mol) - a)
    closest_index = np.argmax(sim_matrix)

    #figure out which node 
    closest_index+=1       #for easy counting
    node_num=0
    for ind,i in enumerate(nodes):
        if closest_index<=i.shape[0]:
            node_num=ind
        else:
            closest_index-=i.shape[0]

    #enter at the node with 'node_num' index
    a = np.dot(nodes[node_num], new_mol)
    sim_matrix = a / (np.sum(nodes[node_num], axis = 1) + np.sum(new_mol) - a)
    closest_index = np.argmax(sim_matrix)

    #find corresponding indices: starting at the k level
    left_indices=[x[0] for x in ns_stats.indices]
    start_left=nodes_id[node_num]+1

    #find the child node to enter through
    for _ in range(closest_index):
        index=bisect_left(left_indices, start_left)
        start_left=ns_stats.indices[index][1]+1

    index=bisect_left(left_indices, start_left)
    start_right=ns_stats.indices[index][1]    #at this point we will have id the child node with start_left and start_right
    print(start_left)

    #traverse down the tree
    while start_right-start_left!=1:
        centroids=ground_up(start_left, start_right, ns_stats)

        a = np.dot(centroids, new_mol)
        sim_matrix = a / (np.sum(centroids, axis = 1) + np.sum(new_mol) - a)
        closest_index = np.argmax(sim_matrix)

        #keep going down child nodes
        for _ in range(closest_index):
            index=bisect_left(left_indices, start_left+1)
            start_left=ns_stats.indices[index][1]+1

            index=bisect_left(left_indices, start_left)
            start_right=ns_stats.indices[index][1]

        start_left=start_left+1
        index=bisect_left(left_indices, start_left)
        start_right=ns_stats.indices[index][1]

        print(start_left)

    #we've reached the leaf level, almost to there
    #identify the leaf node so we can do one more similarity
    index=bisect_left(ns_stats.leaf_idx, start_left)

    a = np.dot(ns_stats.leaf_stats[index][0], new_mol)
    sim_matrix = a / (np.sum(ns_stats.leaf_stats[index][0], axis = 1) + np.sum(new_mol) - a)
    closest_index = np.argmax(sim_matrix)

    final_ls=ns_stats.leaf_stats[0][0][closest_index]
    final_mol_ind=CappedIntBlob(ns_stats.leaf_stats[0][1][closest_index]).lst
    return final_ls, final_mol_ind 





#testings
code=str(CappedIntBlob([1,2,3]))
print(CappedIntBlob(code).lst)

start=time.perf_counter()
ns_stats=build_nested_sets(model)
end=time.perf_counter()
print("time", end-start)

print((ns_stats.indices), len(ns_stats.indices))
ns_stats=reorder_ns_stats_indices(ns_stats)
print(ns_stats.reordered_indices)
print_ns_stats_indices(ns_stats)

start=time.perf_counter()
new_mol=model.root_.subclusters_[1].child_.centroids_[0]
#final_leaf_ls, final_leaf_mol_ind=insert_static(model.root_.subclusters_[1].centroid_, ns_stats)
final_leaf_ls, final_leaf_mol_ind=insert_static_at_k_level(new_mol, 1, ns_stats)

print(final_leaf_mol_ind)

end=time.perf_counter()
print("time", end-start)
#print(final_leaf)


get_pkl_memry(model)
get_pkl_memry(ns_stats)

