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
    for expr in keys(exp_dict[:conditions])
        net_type = exp_dict[:name]
        inpts = exp_dict[:xQ][:inpts]

        fname =  net_type*"_$(expr)_"*make_label_from_keys(inpts)*"_solver"
        f = joinpath("logs",fname)

        info("processing $f")
        d = JLD.load(f*".jld")["problem_dict"]

        exp_json = Dict()

        # variables for making diagnostic plots
        # task set is the set that will be used in the experiment, it is based off of
        # looking at each of the graphs "by hand", and choosing ones that "look good"
        # (i.e. aren't too complicated, truck/motorcycle don't start too close)
        task_set = [100,10,13,16,18,22,23,25,29,30,32,39,41,43,44,48,49,50,54,56,57,58,59,60,62,66,6,74,75,76,77,79,7,80,82,84,87,89,8,90,91,93,94,97,98]
        q = []
        p = []
        oc = []

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
            r_dist = net_vals[:training_data][:r_dist]
            myval = r_dist
            o = rand(r_dist)
            train_mean = net_vals[:training_data][:X3_1]
            train_mean > o ? (outcome = "fail") : (outcome = "success")
            #  info("outcome: $o")
            #  info("train_mean: $train_mean")
            #  info("outcome assessment: $(n[:xP])")
            n[:outcome] = outcome
            outcome == "success" ? oc_bool = 1 : oc_bool = 0
            #  info(n[:outcome])

            if net[1] ∈ task_set
                push!(q,n[:xQ])
                push!(p,n[:xP])
                push!(oc,oc_bool)
            end

            n[:image_file] = joinpath(net_vals[:img_name])
            n[:mcts_depth] = net_vals[:mcts_depth]
        end
        if false
            # make some diagnostic plots to check out the results
            fig,ax = PyPlot.subplots(3,1)
            ax[1][:scatter](q[oc.==1],p[oc.==1],label="success $(length(oc[oc.==1]))")
            ax[1][:scatter](q[oc.==0],p[oc.==0],label="fail $(length(oc[oc.==0]))")
            ax[1][:legend]()
            ax[2][:hist](q[oc.==1])
            ax[2][:hist](q[oc.==0])
            ax[3][:hist](p[oc.==1])
            ax[3][:hist](p[oc.==0],label="p<0 $(length(p[p.<0.]))")
            ax[3][:legend]()
            readline()
        end

        json_fname = "logs/experiment_data_$fname.json"
        info("Writing to $json_fname")
        open(json_fname,"w") do f
            dat = JSON.json(exp_json,4)
            write(f,dat)
        end
    end
end
