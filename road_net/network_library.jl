using LightGraphs, MetaGraphs
using TikzPictures, TikzGraphs

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

    for i in vertices(mg)
        set_prop!(mg,i,:id,i)

        if i == 13
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
function display_network(g::MetaGraph;evader_locs::Array{Int64}=empty!([1]),pursuer_locs::Array{Int64}=empty!([1]),action_locs::Array{Int64}=empty!([1]),fname::String="test",ftype::Symbol=:svg)
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
    t.options = "" # add "scale=2.0" to scale image, but doesn't look too good
    if ftype == :svg
        save(SVG(fname),t)
    else
        println("do not support other outputs yet")
    end
    #  t = plot(g.graph)
    return t
end
