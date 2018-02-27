module Roadnet_MDP

using POMDPs, MetaGraphs, LightGraphs

#  export roadnet_with_pursuer, roadnet_pursuer_state, samenode, rewards, exit_nodes
export roadnet_with_pursuer, roadnet_pursuer_state, rewards, exit_nodes

######### state definition
struct roadnet_pursuer_state
    node::Int64 # current UGV node
    pnode::Int64 # current puruser node
end

function exit_nodes(g::MetaGraph)
    @assert :POMDPgraph in keys(g.gprops)

    return g.gprops[:exit_nodes]
end
function rewards(g::MetaGraph)
    # make sure this is a POMDP MetaGraph -- i.e. one we created with the special metadata
    @assert :POMDPgraph in keys(g.gprops)

    num_vert = nv(g)
    states = Array{roadnet_pursuer_state}(0)
    vals = Array{Float64}(0)
    rwd = g.gprops[:reward_dict]
    # may need to add current believed state in here to account for the reward of hitting the intruder
    for i in vertices(g.graph)
        if g.vprops[i][:state_property] in keys(rwd) # check to see if the state has a reward assigned
            push!(states, roadnet_pursuer_state(i,4))
            push!(vals, rwd[g.vprops[i][:state_property]])
        end
    end
    return states, vals
end

######### MDP definition
struct roadnet_with_pursuer <: MDP{roadnet_pursuer_state,Symbol}
    tprob::Float64 #probability of transitioning to desired state
    discount::Float64 #discount factor
    reward_vals::Vector{Float64}
    reward_states::Vector{roadnet_pursuer_state}
    exit_nodes::Vector{Int64} #list of states that are exits
    road_net::MetaGraph
end

function roadnet_with_pursuer(rn::MetaGraph;tp::Float64=1.0,
                               d::Float64=0.9,
                               rv::Vector{Float64}=rewards(rn)[2],
                               rs::Vector{roadnet_pursuer_state}=rewards(rn)[1],
                               es::Vector{Int64}=exit_nodes(rn))
    return roadnet_with_pursuer(tp,d,rv,rs,es,rn)
end

end
