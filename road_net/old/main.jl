using TikzGraphs, TikzPictures
using LightGraphs, MetaGraphs
using POMDPs
# using Gallium
include("networks.jl")
include("network_viz.jl")
include("bellman.jl")

# Actions are: go to first neighbor, second, etc... this is based on teh order of neighbors in the network representation


function NULL(s)
    #empty for no printing
end

function myMDP(;output=true, its = 5,gamma=1.0)
    mg = ExampleNetwork()
    set_net_props!(mg)

    plot_mg(mg,f_name="original_net")

    # bellman equations
    bellman(mg,output,its,gamma)
    U = get_all_U(mg)
    printU(U)
end

# myMDP(output=true,gamma=0.5, its=55)
myMDP(output=false,gamma=0.9, its=150)
