import JLD, FileIO
import JSON
import PyPlot

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

if run[:json]
    info("## Making .json file for experiment data")
    srand(99) #for consistent results
    for expr in keys(exp_dict[:conditions])
        net_type = exp_dict[:name]
        inpts = exp_dict[:xQ][:inpts]

        fname =  net_type*"_$(expr)_"*make_label_from_keys(inpts)*"_solver"
        f = joinpath("logs",fname)

        info("processing $f")
        d = JLD.load(f*".jld")["problem_dict"]

        exp_json = Dict()

        for net in d
            n = exp_json[net[1]] = Dict()
            net_vals = net[2]

            n[:xQ] = net_vals[:solver_quality][:X_Q]
            n[:xP] = net_vals[:training_data][:X4]

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
                if n[:xQ] > 1.15 && n[:xP] > 0.25
                    # this is an unexpected outcome
                    plot_rwd_dists(net[1],n[:xQ],n[:xP],net_vals[:training_data][:r_dist],net_vals[:solver_quality][:R_star],o,fldr="figs/exp_data/"*net_type*"_$(expr)",fname="$(fname)_$(net[1])",outcome=:fail)
                end
                outcome = "fail"
            else
                if n[:xQ] < 0.75 && n[:xP] < -0.1
                    # this is an unexpected outcome
                    plot_rwd_dists(net[1],n[:xQ],n[:xP],net_vals[:training_data][:r_dist],net_vals[:solver_quality][:R_star],o,fldr="figs/exp_data/"*net_type*"_$(expr)",fname="$(fname)_$(net[1])",outcome=:success)
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
