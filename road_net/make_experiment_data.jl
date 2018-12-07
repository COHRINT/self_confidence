import JLD, FileIO
import JSON
import PyPlot

include("calc_xq.jl")
include("make_roadnet_figs.jl")
include("utilities.jl")

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
    srand(22222) # for consistent results
    # 222 has 21s, 24f
    # 2 has 24 successes and 21 failures, distribution looks OK too
    # 999990 has 21 success and 24 failures, but seems to be lucky when xP is low and xQ is high
    # 999 has 18 successes
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
            if all(r_star_mean .> o) && all(o .< 0.0)
                outcome = "fail"
            else
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
