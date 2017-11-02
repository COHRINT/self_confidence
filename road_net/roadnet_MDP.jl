using LightGraphs, MetaGraphs
using POMDPToolbox

struct roadnetState
    node::Int64
    done::Bool
end

roadnetState(n::Int64) = roadnetState(n,false)
node_equal(s1::roadnetState,s2::roadnetState) = s1.node == s2.node

type roadnet <: MDP{roadnetState,Symbol}
    tprob::Float64 #probability of transitioning to desired state
    discount::Float64 #discount factor
    reward_vals::Vector{Float64}
    reward_states::Vector{roadnetState}
    road_net::MetaGraph
end

function POMDPs.states(mdp::roadnet)
    # return all possible states
    s = Vector{roadnetState}(0)
    for d=0:1, v=1:nv(mdp.road_net)
        push!(s,roadnetState(v,d))
    end
    return s

end

function POMDPs.actions(mdp::roadnet)
    return action_set(mdp)
end

function POMDPs.transition(mdp::roadnet,state::roadnetState,action::Symbol)
    a = action
    n = state.node

    if state.done
        return SparseCat([roadnetState(n,true)],[1.0])
    end

    state_neighbors_idx = neighbors(mdp.road_net,n)
    num_neighbors = length(state_neighbors_idx)
    state_neighbors = states(mdp.road_net,verts=state_neighbors_idx)

    target_idx = get_target(state_neighbors,a)

    probability = fill(0.0,num_neighbors)

    if target_idx == -1
        # action not valid, stay put
        return SparseCat([roadnetState(n)],[1.0])
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

function POMDPs.reward(mdp::roadnet,state::roadnetState,action::Symbol,statep::roadnetState)
    if state.done
        return 0. # already done, no reward left
    end
    r = 0.
    n = length(mdp.reward_states)

    for i = 1:n
        if node_equal(statep,mdp.reward_states[i])
            # if statep is a rewards state add the reward
            r += mdp.reward_vals[i]
        end
    end
    return r
end

POMDPs.n_states(mdp::roadnet) = length(POMDPs.states(mdp)) # gives acces to number of states
POMDPs.n_actions(mdp::roadnet) = length(action_set(mdp)) # gives access to number of actions
POMDPs.discount(mdp::roadnet) = mdp.discount # the discount factor
POMDPs.isterminal(mdp::roadnet, s::roadnetState) = s.done # dictates when MDP solver will terminate

function POMDPs.state_index(mdp::roadnet,state::roadnetState)
    sd = Int(state.done +1)
    idx = sub2ind((nv(mdp.road_net),2),state.node,sd)
    return idx
end

function POMDPs.action_index(mdp::roadnet,act::Symbol)
    @assert act in action_set(mdp)

    return parse(Int,match(r"[0-9]+$",string(act)).match)
end

#### POMDP graph operations

function get_target(neighbors::Vector{roadnetState},a::Symbol)::Int64
    idx = parse(Int,match(r"[0-9]+$",string(a)).match)
    if idx in 1:length(neighbors)
        return idx
    else
        return -1 # this node doesn't have that many neighbors
    end
end
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

function max_d(g::roadnet)
    # make sure this is a POMDP MetaGraph -- i.e. one we created with the special metadata
    @assert :POMDPgraph in keys(g.road_net.gprops)

    return g.road_net.gprops[:net_stats].maximum_degree
end

function action_set(g::roadnet;map::Bool=false)
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
    states = Array{roadnetState}(0)
    vals = Array{Float64}(0)
    rwd = g.gprops[:reward_dict]
    # may need to add current believed state in here to account for the reward of hitting the intruder
    for i in vertices(g.graph)
        if g.vprops[i][:state_property] in keys(rwd) # check to see if the state has a reward assigned
            push!(states, roadnetState(i,isequal(g.vprops[i][:state_property],:exit)))
            push!(vals, rwd[g.vprops[i][:state_property]])
        end
    end
    return states, vals
end
