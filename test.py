import bitbirch.bitbirch as bb
import bitbirch.bitbirch_k as bb_k
import bitbirch.bitbirch_level as bb_level
import bitbirch.bitbirch_kplus1 as bb_kplus1
import bitbirch.bitbirch_kplusn as bb_kplusn 
import bitbirch.bitbirch_entire as bb_entire
import bitbirch.cluster_control as eval
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from rdkit import Chem
from pathlib import Path
from itertools import combinations

def analysis(model, data):
    clusters=eval.birch_analysis(model, data, min_size=0)
    clusters=sorted(clusters, key = lambda x: len(x), reverse = True)

    #evaluate clusters CHI and DBI index
    chi=eval.chi(clusters, reps=False, rep_type="centroid",
                    curated=True, min_size=0)
    dbi=eval.dbi(clusters, reps=False, rep_type="centroid",
                    curated=True, min_size=0)

    #calculate iSIM values for all clusters
    #calculate popultion of each cluster
    #calculate tani between medoid of each cluster pair
    k=len(clusters)
    #print("k",k)
    iSIM=np.zeros(k)
    pop=np.zeros(k)
    medoids=[]

    for ind,i in enumerate(clusters):
        if len(i)==1:
            iSIM[ind]=0
        else:
            iSIM[ind]=eval.jt_isim(np.sum(i,axis=0), i.shape[0])                
        pop[ind]=i.shape[0]
        medoids.append(i[eval.calculate_medoid(i)])

    return [chi, dbi, iSIM, pop, medoids]

def analysis_medoid(medoids1, medoids2):
    medoid_sim=np.zeros((k,k))
    for ind1, i in enumerate(medoids1):
        for ind2, j in enumerate(medoids2):
            medoid_sim[ind1, ind2]=eval.jt_pair(i,j)
    
    return medoid_sim

def cross_validate(fps, fold, size, birch, k):
    avg_chi=0
    avg_dbi=0
    avg_k_isim=np.zeros(k)
    avg_k_pop=np.zeros(k)
    avg_medoids=np.zeros((k,2048))
    avg_time=0

    for i in range(fold):
        rng=np.random.default_rng(i)
        random_indices=rng.choice(fps.shape[0],
                                  size=size,
                                  replace=True)
        data=fps[random_indices]

        birch.set_merge("diameter")
        if birch==bb:
            model=birch.BitBirch(threshold=threshold,
                                 branching_factor=branching_factor)
        else:
            model=birch.BitBirch(threshold=threshold,
                             branching_factor=branching_factor,
                             k=k)
        start_time=time.perf_counter()
        model.fit(data)
        end_time=time.perf_counter()

        '''
        levels=[]
        centroids=[]
        targetLevel=0
        node=model.root_
        model.recursively_traverse(node, levels, centroids, targetLevel)
        print(levels)'''
        


        print(len(model.get_centroids()))
        print("birch ", end_time-start_time)

        
        results=analysis(model, data)
        avg_chi+=results[0]
        avg_dbi+=results[1]
        avg_k_isim+=results[2]
        avg_k_pop+=results[3]
        avg_medoids+=results[4]
        avg_time+=end_time-start_time
        
    
    avg_chi/=fold
    avg_dbi/=fold
    avg_k_isim/=fold
    avg_k_pop/=fold
    avg_medoids/=fold
    avg_time/=fold

    return [avg_chi, avg_dbi, avg_k_isim, avg_k_pop, avg_time, avg_medoids]

def plot_analysis(avg_results):
    for i in range(len(avg_results[0])):        

        k=len(avg_results[0][2])
        x=np.arange(0,k, step=1)
        
        if i==0 or i==1 or i==4:
            fig, ax=plt.subplots()

            values=[res[i] for res in avg_results]
            ax.bar(names, values)
            ax.set_title(metrics[i])
        elif i==2 or i==3:
            fig, ax=plt.subplots()
            
            for j in avg_results:
                ax.plot(x,j[i])
            ax.set_title(metrics[i])
            ax.legend(names)
        else:
            fig, ax=plt.subplots(3,2, figsize=(8,10))
            ax=ax.flatten()
            fig.suptitle(metrics[i])
            
            for ind,j in enumerate(avg_results):
                ax[ind].imshow(j[i])
                ax[ind].set_title(names[ind])
                ax[ind].set_xticks(x)
                ax[ind].set_yticks(x)
            ax[5].axis("off")

            plt.tight_layout()

        plt.savefig(f"results/{metrics[i]}_{sys.argv[1]}_folds_{sys.argv[2]}_size_{sys.argv[3]}_clusters.png",
                    bbox_inches="tight")  
        plt.show()

def plot_medoids(avg_medoids_sim, comb_names):
    fig, ax=plt.subplots(5,2,figsize=(8,10))
    ax=ax.flatten()

    for ind, i in enumerate(avg_medoids_sim):
        ax[ind].set_title(f"{comb_names[ind][0]} and {comb_names[ind][1]}")
        ax[ind].imshow(i)
    plt.tight_layout()
    plt.savefig(f"results/medoid_{sys.argv[1]}_folds_{sys.argv[2]}_size_{sys.argv[3]}_clusters.png")
    plt.show()

folder_path=Path("./results")
folder_path.mkdir(parents=False, exist_ok=True)

'''
with open("chembl.smi","r") as f:
    smiles=np.array([line.strip() for line in f])
#print(smiles.shape[0])

#smiles=smiles[:2000]

mols=[Chem.MolFromSmiles(S) for S in smiles]
fps=np.array([Chem.RDKFingerprint(mol) for mol in mols])
np.save("fingerprints.npy",fps)
'''
fps=np.load("fingerprints.npy")
print("load")

fold=int(sys.argv[1])
size=int(sys.argv[2])
k=int(sys.argv[3])
branching_factor=50
threshold=0.5

birch_list=[bb, bb_level, bb_entire, bb_kplus1, bb_kplusn, bb_k]
avg_results=[]
avg_medoids=[]

for birch in birch_list[1:]:
    results=cross_validate(fps, fold, size, birch, k)
    avg_results.append(results[:5])
    avg_medoids.append(results[5])

names=["bb_level", "bb_entire", "bb_kplusn", "bb_kplus1", "bb_k"]
#names=names[0:2]
metrics=["chi","dbi","iSIM","population_size","time"]

plot_analysis(avg_results)

#analyze medoids
comb=list(combinations(range(2),2))
avg_medoids_sim=[]
comb_names=[]
for i in comb:
    avg_medoids_sim.append(analysis_medoid(avg_medoids[i[0]],avg_medoids[i[1]]))
    comb_names.append((names[i[0]], names[i[1]]))
plot_medoids(avg_medoids_sim, comb_names)






#n=1000
#features=50
#np.random.seed(67)
#data=np.random.randint(2, size=(n,features))
#rng=np.random.default_rng(67)
#data=rng.integers(2, size=(n,features))

'''

# k entire
bb_entire.set_merge('radius')
model_entire=bb_entire.BitBirch(threshold=threshold,
                                branching_factor=branching_factor)
model_entire.fit(data)
#model_entire.trimClusters()
#model_entire.trimClusters2(data)
print("num of leafs entire: ", len(model_entire.get_centroids()))

#cluster sizes
list_mol_ids=model_entire.get_cluster_mol_ids()
for i in list_mol_ids:
    print(len(i),end=" ")
print()





# k+n
bb_kplusn.set_merge('radius')
model_kplusn=bb_kplusn.BitBirch(threshold=threshold,
                                branching_factor=branching_factor,
                                k=10,
                                n=20)
model_kplusn.fit(data)

print("num of leafs k+n: ", len(model_kplusn.get_centroids()))

#cluster sizes
list_mol_ids=model_kplusn.get_cluster_mol_ids()
for i in list_mol_ids:
    print(len(i),end=" ")
print()



# k +1
bb_kplus1.set_merge('radius')
model_k_1=bb_kplus1.BitBirch(threshold=threshold,
                             branching_factor=branching_factor)
model_k_1.fit(data)

print("num of leafs k+1: ", len(model_k_1.get_centroids()))
list_mol_ids=model_k_1.get_cluster_mol_ids()
for i in list_mol_ids:
    print(len(i),end=" ")
print()


# level
bb_level.set_merge('radius')
model_level=bb_level.BitBirch(threshold=threshold,
                              branching_factor=branching_factor)
model_level.fit(data)
model_level.find_level_k()
print("num of leafs_level: ",len(model_level.get_centroids()))
for i in model_level.get_cluster_mol_ids():
    print(len(i), end=" ")
print()


# k leaf clusters
bb_k.set_merge('radius')
model_k=bb_k.BitBirch(threshold=threshold,
                      branching_factor=branching_factor)
model_k.fit(data)

num_leafs_k=len(model_k.get_centroids())
print("num of leafs_k: ",num_leafs_k)

list_mol_ids_k=model_k.get_cluster_mol_ids()
for i in list_mol_ids_k:
    print(len(i),end=" ")
print()
'''





#print(len(model_level.root_.subclusters_))

    