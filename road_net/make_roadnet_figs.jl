using JLD, FileIO
using TikzGraphs
using JSON
using ProgressMeter
include("./network_library.jl")

function make_figs(experiment_dict::Dict,f_loc::String,i_loc::String)
    for expr in keys(experiment_dict[:conditions])
        net_type = experiment_dict[:name]
        fn = experiment_dict[:conditions][expr][:fname]
        fldr = experiment_dict[:conditions][expr][:fldr]

        println("Processing: $fn")
        println("running condition '$expr' data")

        f = joinpath(fldr,fn)

        info("# Running $(f)")
        df = JLD.load("$f.jld")
        d = df["problem_dict"]
        g = df["global_rwd_range"]

        info("Processing Networks")
        p = Progress(length(d),barglyphs=BarGlyphs("[=> ]"),color=:white)
        j = 0
        for net in d
            net_num = net[1]
            net = net[2]
            e_start = net[:evader_start]
            p_start = net[:pursuer_start]
            net_fname = "$(net_type)_$(expr)_net_$net_num"
            f_type = :svg
            display_network(net[:graph],evader_locs=[e_start],pursuer_locs=[p_start],scale=1.0,
                            fname=joinpath(i_loc,net_fname),ftype=f_type)

            net[:img_fldr] = i_loc
            net[:img_name] = net_fname * "." * String(f_type)
            j += 1
            ProgressMeter.next!(p; showvalues = [(:network,"$j of $(length(d))")])
        end

        # re-write original file with "r_dist" object added to each network
        info("Writing to $f")
        jldopen("$f.jld", "w") do file
            write(file, "problem_dict", d)
            write(file, "global_rwd_range", g)
        end
    end
end
