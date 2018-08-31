using JLD
using TikzGraphs
#  Pkg.reload("TikzGraphs")

d = load("logs/x4_test_reference_solver_training.jld")["problem_dict"]

include("./network_library.jl")

q = rand_network(17,exit_nodes=[9],target_mean_degree=2.0)
q2 = rand_network(20,exit_nodes=[9],target_mean_degree=2.0)

#  display_network(d[1][:graph],evader_locs=[1],pursuer_locs=[4])
display_network(q,evader_locs=[1],pursuer_locs=[4],scale=1.0,fname="17net")
display_network(q2,evader_locs=[1],pursuer_locs=[4],scale=1.0,fname="20net")
display_network(d[1][:graph],evader_locs=[1],pursuer_locs=[4],scale=1.0,fname="orig")
