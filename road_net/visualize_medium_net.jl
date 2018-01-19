include("network_library.jl")
#  include("roadnet_pursuer_generator_MDP.jl")

ext_rwd = 2000.
cgt_rwd = -2000.

go = original_roadnet(exit_rwd=ext_rwd,caught_rwd=cgt_rwd,sensor_rwd=-200.)
g = medium_roadnet(exit_rwd=ext_rwd,caught_rwd=cgt_rwd,sensor_rwd=-200.)
#  mdp = roadnet_with_pursuer(g,tp=0.7,d=0.9)

display_network(go,evader_locs=[1],pursuer_locs=[4],fname="original_net",scale=1.0)
display_network(g,evader_locs=[1],pursuer_locs=[4],fname="medium_net",scale=2.0)
