import bitbirch.bitbirch as bb
import bitbirch.bitbirch_partition as bb_part
import bitbirch.cluster_control as eval

import numpy as np
import time
import sys


## merge-triangle inequality
## append-sim threshold

sim_threshold=float(sys.argv[1])
features=2048
threshold=0.4
bf=50
seed_=80

reg_results=np.empty((10,3))
part_results=np.empty((10,5))
filt_results=np.empty((10,3))

np.random.seed(seed_)
for i in range(0,10):
    n=(i+1)*10000
    print(n)
    data=np.random.randint(2, size=(n,features))   

    #regular
    bb.set_merge("radius")
    model=bb.BitBirch(threshold=threshold,
                      branching_factor=bf)
    start=time.perf_counter()
    model.fit(data)
    end=time.perf_counter()

    cluster_analysis=eval.birch_analysis(model, data, min_size=1)
    chi=eval.chi(cluster_analysis)
    dbi=eval.chi(cluster_analysis)

    reg_results[i,0]=chi
    reg_results[i,1]=dbi
    reg_results[i,2]=end-start

    #partition
    bb_part.set_merge("radius")
    model_p=bb_part.BitBirch(threshold=threshold,
                           branching_factor=bf,
                           tolerance=0.005,
                           sim_threshold=sim_threshold)
    start=time.perf_counter()
    model_p.fit(data)
    end=time.perf_counter()

    cluster_analysis=eval.birch_analysis(model_p, data, min_size=1)
    chi=eval.chi(cluster_analysis)
    dbi=eval.chi(cluster_analysis)

    part_results[i,0]=chi
    part_results[i,1]=dbi
    part_results[i,2]=end-start
    part_results[i,3]=len(model_p.noise)/n
    part_results[i,4]=len(model_p.noise)

    #build regular bitbirch with the filtered dataset 
    mask=np.ones(data.shape[0], dtype=bool)
    mask[model_p.noise]=False
    
    filt_data=data[mask, :]
    model=bb.BitBirch(threshold=threshold,
                      branching_factor=bf)
    
    start=time.perf_counter()
    model.fit(filt_data)
    end=time.perf_counter()

    cluster_analysis=eval.birch_analysis(model, filt_data, min_size=1)
    chi=eval.chi(cluster_analysis)
    dbi=eval.chi(cluster_analysis)

    filt_results[i,0]=chi
    filt_results[i,1]=dbi
    filt_results[i,2]=end-start


np.savetxt(f"results_partition/mtas_reg_{features}_{threshold}_{bf}_{seed_}_{sim_threshold}.txt", reg_results)
np.savetxt(f"results_partition/mtas_part_{features}_{threshold}_{bf}_{seed_}_{sim_threshold}.txt", part_results)
np.savetxt(f"results_partition/mtas_filt_{features}_{threshold}_{bf}_{seed_}_{sim_threshold}.txt", filt_results)
