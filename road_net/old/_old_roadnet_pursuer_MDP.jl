using LightGraphs, MetaGraphs
using POMDPs, POMDPToolbox, DiscreteValueIteration
using Distributions
include("network_library.jl")

######### state definition
struct roadnet_pursuer_state
    node::Int64 # current UGV node
    pnode::Int64 # current puruser node
    terminal::Bool #whether this is a termination state or not
end

roadnet_pursuer_state(n::Int64,p::Int64) = roadnet_pursuer_state(n,p,false)
samenode(s1::roadnet_pursuer_state,s2::roadnet_pursuer_state) = s1.node == s2.node


######### MDP definition
type roadnet_with_pursuer <: MDP{roadnet_pursuer_state,Symbol}
    tprob::Float64 #probability of transitioning to desired state
    discount::Float64 #discount factor
    reward_vals::Vector{Float64}
    reward_states::Vector{roadnet_pursuer_state}
    road_net::MetaGraph
end

function roadnet_with_pursuer(rn::MetaGraph;tp::Float64=0.9,
                               d::Float64=0.9,
                               rv::Vector{Float64}=rewards(rn)[2],
                               rs::Vector{roadnet_pursuer_state}=rewards(rn)[1])
    return roadnet_with_pursuer(tp,d,rv,rs,rn)
end

function POMDPs.states(mdp::roadnet_with_pursuer)
    # return all possible states
    s = Vector{roadnet_pursuer_state}(0)
    for d=0:1, v=1:nv(mdp.road_net)
        push!(s,roadnet_pursuer_state(v,d))
    end
    return s

end

function POMDPs.actions(mdp::roadnet_with_pursuer)
    return action_set(mdp)
end

function POMDPs.n_states(mdp::roadnet_with_pursuer)
    return nv(mdp.road_net)^2*2
end

function POMDPs.n_actions(mdp::roadnet_with_pursuer)
    return length(action_set(mdp))
end

function POMDPs.transition(mdp::roadnet_with_pursuer,state::roadnet_pursuer_state,action::Symbol)
    a = action
    n = state.node
    p = state.pnode

    if state.terminal || n==p
        #state is terminal (i.e. exit, or caught)
        return SparseCat([roadnet_pursuer_state(n,p,true)],[1.0])
    end

    ## pursuer move
    pursuer_neighbors_idx = neighbors(mdp.road_net,p)
    pursuer_next_node = pursuer_neighbors_idx[1]::Int #just choose first neighbor for now

    ####### NEED TO FIGURE OUT HOW TO MAKE THE INTRUDER TRANSITION, FIGURING OUT THE STATE PROBABILITY DISTRIBUTION IS GOING TO BE THE DIFFICULT PART
    state_neighbors_idx = neighbors(mdp.road_net,n)::Array{Int}
    num_neighbors = length(state_neighbors_idx)
    state_neighbors = states(mdp.road_net,verts=state_neighbors_idx)

    target_idx = get_target(state_neighbors,a)

    probability = fill(0.0,num_neighbors)

    if target_idx == -1
        # action not valid, stay put
        return SparseCat([roadnet_pursuer_state(n)],[1.0])
    else
        probability[target_idx] = mdp.tprob

        # spread remaining probability to other valid neighbors
        # i.e. spread among remaining neighbors num_neighbors-1
        remaining_prob = (1.0-mdp.tprob)/(num_neighbors-1)
        left_over = collect(1:num_neighbors)

        probability[left_over[left_over.!=target_idx]] = remaining_prob
    end

    return SparseCat(state_neighbors,probability)
end


######### helpers
function states(g::MetaGraph;verts::Array{Int64,1}=collect(vertices(g.graph)))::Vector{roadnetState}
    # make sure this is a POMDP MetaGraph -- i.e. one we created with the special metadata
    @assert :POMDPgraph in keys(g.gprops)

    state_vec = Vector{roadnetState}(0)
    for i in verts
        state_prop = g.vprops[i][:state_property]
        state_bool = false
        if isequal(state_prop,:exit)
            state_bool=true
        end
        push!(state_vec, roadnetState(i,state_bool))
    end

    return state_vec
end

function max_d(g::roadnet_with_pursuer)
    # make sure this is a POMDP MetaGraph -- i.e. one we created with the special metadata
    @assert :POMDPgraph in keys(g.road_net.gprops)

    return g.road_net.gprops[:net_stats].maximum_degree
end

function action_set(g::roadnet_with_pursuer;map::Bool=false)
    # make sure this is a POMDP MetaGraph -- i.e. one we created with the special metadata
    @assert :POMDPgraph in keys(g.road_net.gprops)

    if !map
        return [Symbol("n",i) for i in 1:max_d(g)]
    else
        return collect(1:max_d(g))
    end

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
            push!(states, roadnet_pursuer_state(i,4,isequal(g.vprops[i][:state_property],:exit)))
            push!(vals, rwd[g.vprops[i][:state_property]])
        end
    end
    return states, vals
end
