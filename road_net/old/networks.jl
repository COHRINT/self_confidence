using LightGraphs, MetaGraphs
import LightGraphs.nv, LightGraphs.ne, LightGraphs.vertices
struct network_props
    diam :: Float64
    average_degree :: Float64
    maximum_degree :: Float64
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

# struct my_net_graph
    # g :: MetaGraph
    # vprops :: Dict{Int64,Dict{Symbol,Any}}
    # gprops :: Dict{Symbol,Any}
    # defaultweight :: Float64
    # weightfield :: Symbol
#
    # function my_net_graph(mg::MetaGraph)
        # # should probably calculate, or check for the :U, :U_prime, :transitions, but I'll skip that for now
        # set_net_props!(mg)
        # vprops = mg.vprops
        # gprops = mg.gprops
        # defaultweight = mg.defaultweight
        # weightfield = mg.weightfield
        # new(mg,vprops,gprops)
    # end
# end
#
# function nv(mg::my_net_graph)
    # return nv(mg.g)
# end
# function ne(mg::my_net_graph)
    # return ne(mg.g)
# end
# function vertices(mg::my_net_graph)
    # return vertices(mg.g)
# end
# function vprops(mg::my_net_graph)
    # return vprops(mg.g)
# end
# function gprops(mg::my_net_graph)
    # return gprops(mg.g)
# end
function set_net_props!(mg::MetaGraph)
    set_prop!(mg,:net_stats,network_props(mg))
end

function ExampleNetwork()
    g = Graph(11)
    add_edge!(g,1,2)
    add_edge!(g,1,4)
    add_edge!(g,2,3)
    add_edge!(g,3,5)
    add_edge!(g,5,8)
    add_edge!(g,8,7)
    add_edge!(g,8,11)
    add_edge!(g,11,10)
    add_edge!(g,10,9)
    add_edge!(g,10,7)
    add_edge!(g,6,7)
    add_edge!(g,9,6)
    add_edge!(g,4,6)

    # make graph with metadata
    mg = MetaGraph(g)
    set_prop!(mg,:description, "road_network")

    for i in vertices(mg)
        set_prop!(mg,i,:id,i)
        set_prop!(mg,i,:U,0.0)
        set_prop!(mg,i,:U_prime,0.0)
        set_prop!(mg,i,:Transitions,[])

        if i == 11
            rwd = 1.0
        elseif i == 10
            rwd = -1.0
        else
            rwd = -0.04
        end
        set_prop!(mg,i,:reward,rwd)
    end
    set_net_props!(mg)
    display(mg.gprops)
    return mg
end
