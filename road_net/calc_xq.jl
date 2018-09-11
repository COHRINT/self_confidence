using JLD
using ProgressMeter
using Distributions
using StatsBase
include("make_SQ_model.jl")
include("utilities.jl")
include("self_confidence.jl")

function get_net_input_vals(net::Dict{Symbol,Any},inpts::Array{Symbol})
    info("# Getting inputs")
    ins = Array{Float64}(length(inpts))
    for (i,n) in zip(collect(1:length(inpts)),inpts)
        ins[i] = net[:training_data][n]
    end
    return ins
end

function get_surrogate(fname::String,epocs,fldr::String,net_type::String,inpts::Array{Symbol})
    # make an sq model if it doesn't exist already
    if !(any([contains(x,fname*"_") for x in readdir(fldr)]))
        println("No nn file exists, making one now")
        make_sq_model(net_type,inpts,num_epoc=epocs)
    end

    param_files = searchdir(fldr,fname,".params")
    num_epocs = parse(split(match(r"-\d+",param_files[1]).match,"-")[2])

    SQmodel = load_network(fldr*fname,num_epocs,fldr*fname*"_SQmodel.jld")

    return SQmodel
end

function calculate_sq(experiment_dict)
    for expr in keys(experiment_dict)
        net_type = expr
        inpts = experiment_dict[expr][:inpts]
        epocs = experiment_dict[expr][:epocs]
        cmp_list = experiment_dict[expr][:cmp]
        println("Processing: $(experiment_dict[expr])")
        for cmp in cmp_list
            println("running '$cmp' data")

            ###### Define the variables for the problem
            inputs = Dict()
            for i in inpts
                inputs[i] = "ML.Continuous"
            end

            log_fname = "$(net_type)_$(make_label_from_keys(inputs))"
            log_loc = "nn_logs/"

            ###### get surrogate model
            SQmodel = get_surrogate(log_fname,epocs,log_loc,net_type,inpts)
            info("restoring limits")
            limits = restore_eng_units(SQmodel.range,SQmodel.output_sch)
            println("limits: $limits")
            println("out_sch: $(SQmodel.output_sch)")

            ###### calculate xQ for each net
            f = "logs/$(net_type)_$(cmp)_solver.jld"
            df = JLD.load(f)
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
                ins = get_net_input_vals(net,inpts)

                info("getting predictions")
                println("ins: $ins, $(typeof(ins))")
                _notused, R_star = SQ_predict(SQmodel,ins,use_eng_units=true)
                R_star_μ = R_star[:X3_1][1]
                R_star_σ = R_star[:X3_2][1]

                R_μ = net[:training_data][:X3_1] # mean of rwd dist
                R_σ = net[:training_data][:X3_2] # std of rwd dist
                x_Q, x_Q_Dict = X3(Normal(R_μ,R_σ),Normal(R_star_μ,R_star_σ),global_rwd_range=g,return_raw_sq=true)

                net[:solver_quality] = Dict{Symbol,Any}()
                net[:solver_quality][:x_QDict] = deepcopy(x_Q_Dict)
                net[:solver_quality][:R_star] = R_star
                net[:solver_quality][:X_Q] = x_Q
                j += 1
                ProgressMeter.next!(p; showvalues = [(:network,"$j of $(length(d))")])
            end
            # re-write original file with xQ added to each network
            out_fname = joinpath("logs",net_type*"_"*cmp*"_"*make_label_from_keys(inputs)*"_solver.jld")
            info("# writing to $out_fname")
            jldopen(out_fname, "w") do file
                write(file, "problem_dict", d)
                write(file, "global_rwd_range", g)
            end
        end
    end
end
