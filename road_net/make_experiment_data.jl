import JLD, FileIO
import JSON
import PyPlot
using LightGraphs, MetaGraphs

include("calc_xq.jl")
include("make_roadnet_figs.jl")
include("utilities.jl")
include("plot_rwd_dists.jl")

# make experiment dict
experiment_name = "mturk"
f_loc = "logs"
include("experiment_utilities.jl")

i_loc = "imgs"
myval = []

run = Dict(:figs=>false,:xq=>false,:json=>true)

if run[:figs]
    info("Making Figures")
    make_figs(exp_dict,f_loc,i_loc)
end
if run[:xq]
    info("Calculating xQ")
    # run this second, because we create a new file based on solver to which reference is compared
    calculate_sq(exp_dict)
end

function get_exit_node(g::MetaGraphs.MetaGraph)
    e_nodes = g.gprops[:exit_nodes]
    if length(e_nodes) > 1
        error("something is wrong, should only be length 1 in this experiment")
    else
        return e_nodes[1]
    end
end

if run[:json]
    info("## Making .json file for experiment data")
    srand(5555) #for consistent results
    for expr in keys(exp_dict[:conditions])
        net_type = exp_dict[:name]
        inpts = exp_dict[:xQ][:inpts]

        fname =  net_type*"_$(expr)_"*make_label_from_keys(inpts)*"_solver"
        f = joinpath("logs",fname)

        info("processing $f")
        d = JLD.load(f*".jld")["problem_dict"]

        exp_json = Dict()

        for net in d
            # need to use a HACK because I accidentally placed 13 in the :exit_loc key, when the networks
            # actually have 8 as the exit in the :graph properties. the graph properties are used for
            # the mdp solution, so unless I want to re-create the data (which is expensive in time and money
            # I'm just going to write some hack code to get the exit node from the graph
            exit_loc = get_exit_node(net[2][:graph])
            dist_to_exit = length(a_star(net[2][:graph],net[2][:evader_start],exit_loc))
            dist_to_pursuer = length(a_star(net[2][:graph],net[2][:evader_start],net[2][:pursuer_start]))
            edge_to_node_ratio = ne(net[2][:graph])/nv(net[2][:graph])
            etnr = 2.5
            dte = 2.0
            dtp = 2.0
            if dist_to_exit < dte || dist_to_pursuer < dtp || edge_to_node_ratio > etnr
                # we don't want networks where exit or pursuer is too close at the beginning
                # or where the edge/node ratio is too high (i.e. too many edges)
                if dist_to_exit < dte
                    println("Exit too close (d=$dist_to_exit): skipping $(net[1])")
                end
                if dist_to_pursuer < dtp
                    println("Pursuer too close (d=$dist_to_pursuer): skipping $(net[1])")
                end
                if edge_to_node_ratio > etnr
                    println("Too many edges ($edge_to_node_ratio): skipping $(net[1])")
                end
                continue
            end

            n = exp_json[net[1]] = Dict()
            net_vals = net[2]

            n[:xQ] = net_vals[:solver_quality][:X_Q]
            n[:xP] = net_vals[:training_data][:X4]
            n[:dist_to_exit] = dist_to_exit
            n[:dist_to_pursuer] = dist_to_pursuer
            n[:edge_to_node_ratio] = edge_to_node_ratio

            # calculate whether experiment succeded or not, take reward distribution and draw at random
            # this will have a higher probability of drawing most repeated numbers, so it is like sampling
            # a parameterized distribution, with the limitaion that it will only draw an outcome that has
            # already been seen
            # If that value is greater than the mean expected reward of the trusted distribution (:X3_1) then the experiment is a success
            # otherwise it is a failure
            r_dist = net_vals[:training_data][:r_dist]
            myval = r_dist
            o = rand(r_dist)

            #  r_star_mean = net_vals[:training_data][:X3_1] # Old, WRONG way, leave hear for illustrative purposes
            r_star_mean = net_vals[:solver_quality][:R_star][:X3_1]
            #  if all(r_star_mean .> o) #old WRONG way
            #  if all(r_star_mean .> o) && all(o .< 0.0)
            if all(o .< 0.0)
                if n[:xQ] < 0.5 && n[:xP] > 0.5
                    # this is an unexpected outcome
                    plot_rwd_dists(net[1],n[:xQ],n[:xP],net_vals[:training_data][:r_dist],net_vals[:solver_quality][:R_star],o,fldr="figs/exp_data/"*net_type*"_$(expr)",fname="$(fname)_$(net[1])_bigfail",outcome=:fail)
                end
                if n[:xQ] > 1.15 && n[:xP] > 0.25
                    # this is an unexpected outcome
                    plot_rwd_dists(net[1],n[:xQ],n[:xP],net_vals[:training_data][:r_dist],net_vals[:solver_quality][:R_star],o,fldr="figs/exp_data/"*net_type*"_$(expr)",fname="$(fname)_$(net[1])_surprisefail",outcome=:fail)
                end
                outcome = "fail"
            else
                if n[:xQ] < 0.75 && n[:xP] < -0.1
                    # this is an unexpected outcome
                    plot_rwd_dists(net[1],n[:xQ],n[:xP],net_vals[:training_data][:r_dist],net_vals[:solver_quality][:R_star],o,fldr="figs/exp_data/"*net_type*"_$(expr)",fname="$(fname)_$(net[1])_surprisesuccess",outcome=:success)
                end
                outcome = "success"
            end
            n[:outcome] = outcome
            n[:num_nodes] = net_vals[:num_nodes]
            n[:trans_prob] = net_vals[:trans_prob]
            n[:rwd_draw] = o
            n[:trusted_mean] = r_star_mean
            outcome == "success" ? oc_bool = 1 : oc_bool = 0
            #  println("rand draw: $o")
            #  println("trusted mean: $r_star_mean")
            #  println("oucome: $outcome")
            #  readline()
            #  info(n[:outcome])

            n[:image_file] = joinpath(net_vals[:img_name])
            n[:mcts_depth] = net_vals[:mcts_depth]
        end

        json_fname = "logs/experiment_data_$(fname).json"
        info("Writing to $json_fname")
        open(json_fname,"w") do f
            dat = JSON.json(exp_json,4)
            write(f,dat)
        end
    end
end
