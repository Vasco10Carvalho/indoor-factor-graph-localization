include "cartographer_2d_comparison.lua"

-- The trajectory loaded from the pbstream is frozen and serves as the map.
-- Only a small rolling set of submaps from the current run is retained.
TRAJECTORY_BUILDER.pure_localization_trimmer = {
  max_submaps_to_keep = 3,
}

-- Pure localization needs frequent global corrections, but aggressive global
-- sampling would be too expensive against every frozen submap.
POSE_GRAPH.optimize_every_n_nodes = 10
POSE_GRAPH.global_sampling_ratio = 0.003
POSE_GRAPH.constraint_builder.sampling_ratio = 0.03

return options
