import numpy as np



stats = [[1.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0 ,   0.0   ],
 [0.986, 0.989, 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ],
 [0.976, 0.988, 0.971, 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ],
 [0.973, 0.984, 0.943, 0.965, 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ],
 [0.972, 0.984, 0.911, 0.941, 0.954, 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ],
 [0.972, 0.985, 0.906, 0.825, 0.95 , 0.901, 0.0 , 0.0 , 0.0 , 0.0 ],
 [0.97 , 0.984, 0.894, 0.819, 0.938, 0.893, 0.946, 0.0 , 0.0 , 0.0 ],
 [0.967, 0.984, 0.892, 0.814, 0.915, 0.886, 0.946, 0.949, 0.0 , 0.0 ],
 [0.933, 0.977, 0.893, 0.812, 0.915, 0.886, 0.946, 0.949, 0.967, 0.0 ],
 [0.925, 0.944, 0.892, 0.809, 0.916, 0.886, 0.945, 0.949, 0.962, 0.924]]

stats = np.array(stats)


all_avg_stats = []
for stage_i in range(len(stats)):
    
    stage_stats = stats[stage_i][0:stage_i+1]
   
    avg_stats_of_stage = np.mean(stage_stats)
    all_avg_stats.append(avg_stats_of_stage)
    print(f"Stage {stage_i} average: {avg_stats_of_stage:.2f}")

print(f"Average stats of all stages: {np.mean(all_avg_stats)}")
print(f"Final average stats: {np.mean(stats[-1])}")









