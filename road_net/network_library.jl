using LightGraphs, MetaGraphs

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

function original_roadnet()::MetaGraph
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
    set_prop!(mg,:POMDPgraph,true) # this indicates that we created this graph with POMDP structure
    set_prop!(mg,:description, "original road network")
    set_prop!(mg,:reward_dict, Dict([(:exit,1000.),(:caught,-1000.),(:sensor,-1.)]))
    # set_prop!(mg,:reward_dict, Dict([(:exit,1000.),(:caught,-1000.)]))

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
