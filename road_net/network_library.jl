using LightGraphs, MetaGraphs
using TikzPictures, TikzGraphs
using DataStructures

struct network_props
    # structure to handle network properties, these are used to train
    # an EHM-style model. That is, they are used to train a model to
    # predict POMDP solver quality based off of the graph properties...
    # at least theoretically :-)
    diam :: Int64
    average_degree :: Float64
    maximum_degree :: Int64
    degree_variance :: Float64
    nv :: Int64
    ne :: Int64

    # Default constructor
    function network_props(mg::MetaGraph)
        diam = diameter(mg)
        deg_list = degree(mg)
        average_degree = mean(deg_list)
        maximum_degree = maximum(deg_list)
        degree_variance = var(deg_list)
        nv = MetaGraphs.nv(mg)
        ne = MetaGraphs.ne(mg)
        new(diam,average_degree,maximum_degree,degree_variance,nv,ne)
    end
end

function set_net_props!(mg::MetaGraph)
    set_prop!(mg,:net_stats,network_props(mg))
end


#### Network Library

function original_roadnet(;exit_rwd::Float64=1000.,caught_rwd::Float64=-2000.,sensor_rwd::Float64=-1.)::MetaGraph
    g = MetaGraph(13)
    add_edge!(g,1,2)
    add_edge!(g,1,10)
    add_edge!(g,2,3)
    add_edge!(g,2,9)
    add_edge!(g,3,4)
    add_edge!(g,4,5)
    add_edge!(g,4,7)
    add_edge!(g,5,6)
    add_edge!(g,6,7)
    add_edge!(g,6,8)
    add_edge!(g,7,8)
    add_edge!(g,7,11)
    add_edge!(g,8,12)
    add_edge!(g,8,13)
    add_edge!(g,9,10)
    add_edge!(g,9,11)
    add_edge!(g,10,12)
    add_edge!(g,11,12)
    add_edge!(g,12,13)

    # make graph with metadata
    mg = MetaGraph(g)
    set_prop!(mg,:POMDPgraph,true) # this indicates that we created this graph with POMDP structure (as defined by me)
    set_prop!(mg,:description, "original road network")
    set_prop!(mg,:reward_dict, Dict([(:exit,exit_rwd),(:caught,caught_rwd),(:sensor,sensor_rwd)]))
    set_prop!(mg,:exit_nodes,[13])

    for i in vertices(mg)
        set_prop!(mg,i,:id,i)

        if i in mg.gprops[:exit_nodes]
            state_prop = :exit
        else
            state_prop = :sensor
        end
        set_prop!(mg,i,:state_property,state_prop)
    end
    set_net_props!(mg)
    return mg
end

function medium_roadnet(;exit_rwd::Float64=1000.,caught_rwd::Float64=-2000.,sensor_rwd::Float64=-1.)::MetaGraph
    srand(2111) #fix the rng seed for reproducibility
    g_connected = false
    g = Graph()
    while !g_connected
        g = erdos_renyi(45,0.05) #aiming for max degree of ~5
        g_connected = is_connected(g)
    end

    # make graph with metadata
    mg = MetaGraph(g)
    set_prop!(mg,:POMDPgraph,true) # this indicates that we created this graph with POMDP structure (as defined by me)
    set_prop!(mg,:description, "medium sized road network")
    set_prop!(mg,:reward_dict, Dict([(:exit,exit_rwd),(:caught,caught_rwd),(:sensor,sensor_rwd)]))
    set_prop!(mg,:exit_nodes,[32])

    for i in vertices(mg)
        set_prop!(mg,i,:id,i)

        if i in mg.gprops[:exit_nodes]
            state_prop = :exit
        else
            state_prop = :sensor
        end
        set_prop!(mg,i,:state_property,state_prop)
    end
    set_net_props!(mg)
    return mg
end
###### add other road networks here.....
function rand_network(n::Int64;exit_rwd::Float64=1000.,caught_rwd::Float64=-2000.,
                      sensor_rwd::Float64=-1.,net_seed::Int64=0,
                      target_mean_degree::Float64=5.0,is_directed::Bool=false,
                      exit_nodes::Array{Int64}=empty!([1]))::MetaGraph
    # n: total nodes
    # exit_rwd: how much reward at exit
    # caught_rwd: how much reward when caught
    # sensor_rwd: baseline reward, incentive to keep moving instead of staying put
    # net_seed: random seed to be able to regenerate a network
    # target_mean_degree: desired average degree to constrain the action space
    # is_directed: whether graph is directed or not

    g = Graph()
    p = 0.5
    p_itr = 1
    tot_itr = 1
    max_its = 1e4
    round_digits = length(split(string(target_mean_degree),".")[2])
    d_hist = Deque{Float64}()
    while (!is_connected(g) || round(mean(degree(g)),round_digits)!= target_mean_degree) && tot_itr < max_its
        g = erdos_renyi(n,p,is_directed=is_directed,seed=net_seed)
        tot_itr += 1
        p_itr += 1
        push!(d_hist,mean(degree(g))) #keep running history of mean degrees
        if p_itr > 100
            # we aren't getting desired degree, change p accordingly
            p_delta = 10.^(-(log(p_itr)-2.))
            if mean(degree(g)) > target_mean_degree
                # subtract less and less off of p as iterations go up
                p = p - p_delta
            else
                # add less and less to p as iterations go up
                p = p + p_delta
            end
            shift!(d_hist)
            #  round(p,5)
            #  println("Mean degree is $(mean(degree(g))), target is $(target_mean_degree), Changing p to: $p")
        end
    end
    if tot_itr >= max_its
        d_hist_diff = abs.(diff([float(x) for x in d_hist]))
        #  display(d_hist)
            #  display(d_hist_diff)
            #  display(unique(d_hist_diff))
            if length(unique(d_hist_diff)) > 3
                    # if there are "too many" unique values in history, couldn't find it
                    # otherwise were were bouncing around the solution, and can pass on the answer
            error("couldn't make desired network")
        end
    end

    # make graph with metadata
    mg = MetaGraph(g)
    set_prop!(mg,:POMDPgraph,true) # this indicates that we created this graph with POMDP structure (as defined by me)
    set_prop!(mg,:description, "automatically generated random network with n=$n, and p=$p, and targeted average degree=$target_mean_degree")
    set_prop!(mg,:reward_dict, Dict([(:exit,exit_rwd),(:caught,caught_rwd),(:sensor,sensor_rwd)]))
    set_prop!(mg,:exit_nodes,exit_nodes)

    for i in vertices(mg)
        set_prop!(mg,i,:id,i)

        if i in mg.gprops[:exit_nodes]
            state_prop = :exit
        else
            state_prop = :sensor
        end
        set_prop!(mg,i,:state_property,state_prop)
    end
    set_net_props!(mg)
    return mg
end

##### display stuff
function display_network(g::MetaGraph;evader_locs::Array{Int64}=empty!([1]),pursuer_locs::Array{Int64}=empty!([1]),action_locs::Array{Int64}=empty!([1]),fname::String="test",ftype::Symbol=:svg,scale::Float64=1.0)
    # find exit nodes
    node_styles = Dict()
    exit_format = "fill=green!70"
    other_node_style = "draw, rounded corners, fill=blue!10"
    for i in vertices(g)
        if g.vprops[i][:state_property] == :exit
            node_styles[i]=exit_format
        end
    end

    evader_node_style = "fill=yellow!70"
    action_node_style = "fill=yellow!30"
    pursuer_node_style = "fill=red!70"
    for x in action_locs
        node_styles[x] = action_node_style
    end
    for x in evader_locs
        node_styles[x] = evader_node_style
    end
    for x in pursuer_locs
        node_styles[x] = pursuer_node_style
    end

    t = TikzGraphs.plot(g.graph,Layouts.Spring(randomSeed=52), node_style=other_node_style,node_styles=node_styles)
    t.options = "scale=$scale" # add "scale=2.0" to scale image, but doesn't look too good
    if ftype == :svg
        TikzPictures.save(SVG(fname),t)
    else
        println("do not support other outputs yet")
    end
    #  t = plot(g.graph)
    return t
end
