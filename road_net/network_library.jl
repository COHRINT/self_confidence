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
function rand_network(N::Int64;exit_rwd::Float64=1000.,caught_rwd::Float64=-2000.,
                      sensor_rwd::Float64=-1.,net_seed::Int64=0,
                      target_mean_degree::Float64=5.0,approx_E::Int64=2*N,is_directed::Bool=false,
                      exit_nodes::Array{Int64}=empty!([1]),method::Symbol=:erdos_n_p)::MetaGraph
    # N: total nodes
    # exit_rwd: how much reward at exit
    # caught_rwd: how much reward when caught
    # sensor_rwd: baseline reward, incentive to keep moving instead of staying put
    # net_seed: random seed to be able to regenerate a network
    # target_mean_degree: desired average degree to constrain the action space, if using method=erdos_n_p
    # approx_E: approximate number of edges to be used if using method=:erdos_n_e (some randomness added here)
    # is_directed: whether graph is directed or not
    # method: specifies using erdos_n_p or erdos_n_e

    g = Graph()
    p_itr = 1
    tot_itr = 1
    max_its = 1e4
    round_digits = length(split(string(target_mean_degree),".")[2])
    d_hist = Deque{Float64}()
    if method == :erdos_n_p
        #  p = 0.5
        approx_desired_edges = N*target_mean_degree*2.
        p = N./approx_desired_edges
        println("##########$p##########")
        while (!is_connected(g) || round(mean(degree(g)),round_digits)!= target_mean_degree) && tot_itr < max_its
            g = erdos_renyi(N,p,is_directed=is_directed,seed=net_seed)
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
    elseif method == :erdos_n_e
        while (!is_connected(g)) && tot_itr < max_its
            E = Int(round(approx_E + randn()/10*approx_E))
            g = erdos_renyi(N,E,is_directed=is_directed,seed=net_seed)
            net_seed += 1
        end
    else
        error("DO NOT SUPPORT THAT METHOD YET")
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
    set_prop!(mg,:description, "automatically generated random network with n=$N, using method: $(string(method))")
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
function display_network(g::MetaGraph;evader_locs::Array{Int64}=empty!([1]),pursuer_locs::Array{Int64}=empty!([1]),action_locs::Array{Int64}=empty!([1]),fname::String="test",ftype::Symbol=:pdf,scale::Float64=1.0,exit_node_style::String="fill=white!70,inner sep=1pt",evader_node_style::String="fill=teal,inner sep=1pt",action_node_style::String="fill=yellow!30",pursuer_node_style::String="fill=red!70,inner sep=1pt",other_node_style::String="circle, fill=lightgray",label_style::Symbol=:debug,evader_font_color::String="black",pursuer_font_color::String="black",exit_font_color::String="black")
    # find exit nodes
    node_styles = Dict()
    node_labels = Vector{AbstractString}(length(vertices(g)))
    for i in vertices(g)
        if g.vprops[i][:state_property] == :exit
            node_styles[i] = exit_node_style
            #  node_labels[i] = LaTeXString("\\Stopsign") # marvosym package
            #  node_labels[i] = LaTeXString("\\FiveStarOpenCircle") # bbding package
            node_labels[i] = LaTeXString("\{\\color{$(exit_font_color)}\\faStar\}") # fontawesome package
        else
            node_labels[i] = ""
        end
    end

    for x in action_locs
        node_styles[x] = action_node_style
    end
    for x in evader_locs
        node_styles[x] = evader_node_style
        #  node_labels[x] = LaTeXString("\\Smiley") # marvosym package
        #  node_labels[x] = LaTeXString("\\faPlane") # fontawesome package
        node_labels[x] = LaTeXString("\{\\color{$(evader_font_color)}\\faTruck\}") # fontawesome package
    end
    for x in pursuer_locs
        node_styles[x] = pursuer_node_style
        #  node_labels[x] = LaTeXString("\\Frowny")
        node_labels[x] = LaTeXString("\{\\color{$(pursuer_font_color)}\\faMotorcycle\}") # fontawesome package

    end
    #  prmbl = "\\usepackage{fontawesome}\n\\usepackage{pifont}\n\\usepackage{textcomp}\n\\DeclareFontFamily\{U\}\{magic\}\{\}\n"
    prmbl = readstring("./prepend_preamble.tex")

    #  t = TikzGraphs.plot(g.graph,labels=node_labels,Layouts.Layered(), node_style=other_node_style,node_styles=node_styles,prepend_preamble=prmbl)
    t = TikzGraphs.plot(g.graph,labels=node_labels,Layouts.SpringElectrical(randomSeed=100,charge=5.), node_style=other_node_style,node_styles=node_styles,prepend_preamble=prmbl)
    t.options = "scale=$scale" # add "scale=2.0" to scale image, but doesn't look too good
    if ftype == :pdf
        TikzPictures.save(PDF(fname),t)
    else
        println("do not support other outputs yet")
    end
    #  t = plot(g.graph)
    return t
end
