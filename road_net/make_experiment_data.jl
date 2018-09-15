import JLD, FileIO
import JSON

include("calc_xq.jl")
include("make_roadnet_figs.jl")
include("utilities.jl")

# make experiment dict
experiment_name = "mturk_fast"
f_loc = "logs"
include("experiment_utilities.jl")

i_loc = "imgs"

run = Dict(:figs=>true,:xq=>false,:json=>false)

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
            # If that value is greater than the mean expected reward (:X3_1) then the experiment is a success
            # otherwise it is a failure
            o = rand(net_vals[:training_data][:r_dist])
            net_vals[:training_data][:X3_1] > o ? (outcome = "success") : (outcome = "fail")
            n[:outcome] = outcome

            n[:image_file] = joinpath(net_vals[:img_fldr],net_vals[:img_name])
            n[:mcts_depth] = net_vals[:mcts_depth]
        end
        json_fname = "logs/experiment_data_$fname.json"
        info("Writing to $json_fname")
        open(json_fname,"w") do f
            dat = JSON.json(exp_json,4)
            write(f,dat)
        end
    end
end
