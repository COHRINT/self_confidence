using POMDPs, BasicPOMCP, POMDPToolbox
using MetaGraphs
using Distributions, StatsBase
include("network_library.jl")

######### state definition
struct roadnet_pursuer_state
    node::Int64 # current UGV node
    pnode::Int64 # current puruser node
end

samenode(s1::roadnet_pursuer_state,s2::roadnet_pursuer_state) = s1.node == s2.node && s1.pnode == s1.pnode


######### MDP definition
type roadnet_with_pursuer <: MDP{roadnet_pursuer_state,Symbol}
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

######### MDP common definitions
POMDPs.n_states(mdp::roadnet_with_pursuer) = length(POMDPs.states(mdp)) # gives acces to number of states
POMDPs.n_actions(mdp::roadnet_with_pursuer) = length(action_set(mdp)) # gives access to number of actions
POMDPs.discount(mdp::roadnet_with_pursuer) = mdp.discount # the discount factor
POMDPs.isterminal(mdp::roadnet_with_pursuer, s::roadnet_pursuer_state) = s.node in mdp.exit_nodes || s.node==s.pnode # dictates when MDP solver will terminate
POMDPs.initial_state(mdp::roadnet_with_pursuer) = roadnet_with_pursuer(1,10) #could be a distribution, but we'll fix it for now

function POMDPs.states(mdp::roadnet_with_pursuer)
    # return all possible states
    s = Vector{roadnet_pursuer_state}(0)
    for v=1:nv(mdp.road_net), p=1:nv(mdp.road_net)
        push!(s,roadnet_pursuer_state(v,p))
    end
    return s
end

function POMDPs.state_index(mdp::roadnet_with_pursuer,state::roadnet_pursuer_state)
    idx = sub2ind((nv(mdp.road_net),2),state.node,state.pnode)
    return idx
end

function POMDPs.actions(mdp::roadnet_with_pursuer)
    return action_set(mdp)
end

function POMDPs.action_index(mdp::roadnet_with_pursuer,act::Symbol)
    @assert act in action_set(mdp)

    return parse(Int,match(r"[0-9]+$",string(act)).match)
end

function POMDPs.generate_s(mdp::roadnet_with_pursuer,state::roadnet_pursuer_state,action::Symbol,rng::AbstractRNG)
    # make sure this is a POMDP MetaGraph -- i.e. one we created with the special metadata
    @assert :POMDPgraph in keys(mdp.road_net.gprops)

    # generate next UGV state
    a = action
    n = state.node
    p = state.pnode

    ## pursuer move
    ## pursuer doesn't have uncertainty in movements, there is no selected action, the pursuer state just updates according to a specified method
    pursuer_neighbors_idx = neighbors(mdp.road_net,p)
    newPursuer_node = pursuer_movement(pursuer_neighbors_idx,p,method=:random)
    #  newPursuer_node = pursuer_neighbors_idx[1]::Int #just choose first neighbor for now

    ## UGV move
    state_neighbors_idx = neighbors(mdp.road_net,n)
    num_neighbors = length(state_neighbors_idx)

    target_idx = get_target(state_neighbors_idx,a)

    UGV_probability = fill(0.0,num_neighbors)

    # make transition probability
    if target_idx == -1
        # action not valid, stay put
        return roadnet_pursuer_state(state.node,newPursuer_node)
    else
        UGV_probability[target_idx] = mdp.tprob

        # spread remaining probability to other valid neighbors
        # i.e. spread among remaining neighbors num_neighbors-1
        remaining_prob = (1.0-mdp.tprob)/(num_neighbors-1)
        left_over = collect(1:num_neighbors)

        UGV_probability[left_over[left_over.!=target_idx]] = remaining_prob
    end

    newUGV_node = sample(state_neighbors_idx,Weights(UGV_probability))

    exit = is_exit(mdp,newUGV_node)

    return roadnet_pursuer_state(newUGV_node,newPursuer_node)

end

function POMDPs.reward(mdp::roadnet_with_pursuer,state::roadnet_pursuer_state,action::Symbol,statep::roadnet_pursuer_state)
    #  println(mdp.road_net.gprops[:exit_nodes])
    if state.node in mdp.road_net.gprops[:exit_nodes]
        # reached exit
        #  println("#####escaped######")
        return mdp.reward_vals[state.node]
    end
    if statep.node == statep.pnode
        # we were aprehended
        #  println("####aprehended#####")
        #  println(mdp.road_net.gprops[:reward_dict][:caught])
        return mdp.road_net.gprops[:reward_dict][:caught]
    end

    r = 0.
    n = length(mdp.reward_states)
    # otherwise look up the reward value
    for i = 1:n
        if samenode(statep,mdp.reward_states[i])
            # if statep is a rewards state add the reward
            r += mdp.reward_vals[i]
            #  if r == 1000.
                #  println("escaped")
            #  elseif r == 2000.
                #  println("caught")
            #  else
                #  println("moved")
            #  end
        end
    end
    return r
end

#  function POMDPs.initial_state_distribution(p::roadnet_with_pursuer)
    #  return roadnet_pursuer_state(1,3)
#  end

##### helper functions
function is_exit(g::roadnet_with_pursuer,u::Int64)
    return g.road_net.vprops[u][:state_property] == :exit
end

function pursuer_movement(n::Array{Int64},p::Int64;method::Symbol=:random)::Int64
    if method == :random
        # random walk, allow pursuer to also stay put
        # TODO, pursuer and ugv can avoid collision if they move to each other's nodes simultaneously, maybe fix this?
        # TODO, pursuer can occupy the exit, possibly fix at some point
        n_copy = copy(n)
        push!(n_copy,p)
        q = sample(n_copy)
        #  println("p: $p, n: $n, n_copy: $n_copy, q: $q")
        return q
    elseif method == :first
        # choose the first neighbor
        return n[1]
    else
        println("Method not yet supported")
    end
end
function max_d(g::roadnet_with_pursuer)
    # make sure this is a POMDP MetaGraph -- i.e. one we created with the special metadata
    @assert :POMDPgraph in keys(g.road_net.gprops)

    return g.road_net.gprops[:net_stats].maximum_degree
end

function get_target_node(g::MetaGraph,n::Int64,a::Symbol)::Int64
    # make sure this is a POMDP MetaGraph -- i.e. one we created with the special metadata
    @assert :POMDPgraph in keys(g.gprops)

    node_neighbors = neighbors(g.graph,n)
    target = get_target(neighbors(g,n),a)

    return node_neighbors[target]
end

function get_target(neighbors::Vector{Int64},a::Symbol)::Int64
    idx = parse(Int,match(r"[0-9]+$",string(a)).match)
    if idx in 1:length(neighbors)
        return idx
    else
        return -1 # this node doesn't have that many neighbors
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
            push!(states, roadnet_pursuer_state(i,4))
            push!(vals, rwd[g.vprops[i][:state_property]])
        end
    end
    return states, vals
end

function exit_nodes(g::MetaGraph)
    @assert :POMDPgraph in keys(g.gprops)

    return g.gprops[:exit_nodes]
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

# MCTS node labels
function MCTS.node_tag(s::roadnet_pursuer_state)
    if s.exit
        return "EXIT"
    elseif s.node == s.pnode
        return "CAUGHT"
    else
        return "[$(s.node),$(s.pnode)]"
    end
end

MCTS.node_tag(a::Symbol) = "[$a]"
