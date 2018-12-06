using JLD
using ProgressMeter
using Distributions
using StatsBase
include("make_SQ_model.jl")
include("utilities.jl")
include("self_confidence.jl")

function get_net_input_vals(net::Dict{Symbol,Any},inpts::Array{Symbol},sch::ML.Schema;return_scaled::Bool=false)
    info("# Getting inputs")
    ins = Array{Float64}(length(inpts),1)
    scaled_ins = Array{Float64}(length(inpts),1)
    for (i,n) in zip(collect(1:length(inpts)),inpts)
        ins[i] = net[:training_data][n]
        if return_scaled
            scaled_ins[i] = (ins[i]-mean(sch[n]))/std(sch[n])
        end
    end
    if return_scaled
        return ins, scaled_ins
    else
        return ins
    end
end

function get_surrogate(fname::String,epocs::Int64,fldr::String,net_type::String,
                       inpts::Array{Symbol},trusted_fname::String)
    # make an sq model if it doesn't exist already
    if !(any([contains(x,fname*"_") for x in readdir(fldr)]))
        info("No nn file exists, making one now")
        make_sq_model(net_type,inpts,num_epoc=epocs,trusted_fname=trusted_fname)
    end

    info("Loading xQ Model")
    param_files = searchdir(fldr,fname,".params")
    num_epocs = parse(split(match(r"-\d+",param_files[1]).match,"-")[2])

    fn = joinpath(fldr,fname)
    SQmodel = load_network(fn,num_epocs,fn*"_SQmodel.jld")

    return SQmodel
end

function calculate_sq(experiment_dict)
    trusted_cond = 0
    for c in keys(experiment_dict[:conditions])
        if experiment_dict[:conditions][c][:role] == :trusted
            trusted_cond = c
        end
    end
    for expr in keys(experiment_dict[:conditions])
        net_type = experiment_dict[:name]
        inpts = experiment_dict[:xQ][:inpts]
        epocs = experiment_dict[:xQ][:epocs]
        println("Processing: $(experiment_dict[:conditions][expr])")

        println("running '$expr' data")

        ###### Define the variables for the problem
        inputs = Dict()
        for i in inpts
            inputs[i] = "ML.Continuous"
        end

        log_fname = "$(net_type)_$(make_label_from_keys(inputs))"
        log_loc = experiment_dict[:xQ][:nn_loc]
        tfname = experiment_dict[:conditions][trusted_cond][:fname]
        tfold = experiment_dict[:conditions][trusted_cond][:fldr]
        cfname = experiment_dict[:conditions][expr][:fname]
        cfold = experiment_dict[:conditions][expr][:fldr]
        trusted_path = joinpath(tfold,tfname*".csv")

        ###### get surrogate model
        SQmodel = get_surrogate(log_fname,epocs,log_loc,net_type,inpts,trusted_path)
        info("restoring limits")
        limits = restore_eng_units(SQmodel.range,SQmodel.output_sch)
        println("limits: $limits")
        println("out_sch: $(SQmodel.output_sch)")

        ###### calculate xQ for each net
        df = JLD.load(joinpath(cfold,cfname*".jld"))
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
            ins, scaled_ins = get_net_input_vals(net,inpts,SQmodel.input_sch,return_scaled=true)
            #  println(ins)
            #  println(scaled_ins)

            info("getting predictions")
            println("ins: $ins, $(typeof(ins))")
            _notused, R_star = SQ_predict(SQmodel,scaled_ins,use_eng_units=true)
            R_star_μ = R_star[:X3_1][1]
            R_star_σ = R_star[:X3_2][1]

            R_μ = net[:training_data][:X3_1] # mean of rwd dist
            R_σ = net[:training_data][:X3_2] # std of rwd dist

            info("R_star_μ = $R_star_μ, R_star_σ = $R_star_σ")
            info("R_μ = $R_μ, R_σ = $R_σ")
            # make sure sigma isn't zero or negative (i.e. model could have bad prediction, or data could be all an identical number)
            #  R_star_σ <= 0. ? error() : nothing
            #  R_σ <= 0. ? error() : nothing
            R_star_σ <= 0. ? R_star_σ = 1. : nothing
            R_σ <= 0. ? R_σ = 1. : nothing
            #  error()

            x_Q, x_Q_Dict = X3(Normal(R_μ,R_σ),Normal(R_star_μ,R_star_σ),global_rwd_range=g,return_raw_sq=true)

            net[:solver_quality] = Dict{Symbol,Any}()
            net[:solver_quality][:x_QDict] = deepcopy(x_Q_Dict)
            net[:solver_quality][:R_star] = R_star
            net[:solver_quality][:X_Q] = x_Q
            j += 1
            ProgressMeter.next!(p; showvalues = [(:network,"$j of $(length(d))")])
        end
        # re-write original file with xQ added to each network
        out_fname = joinpath("logs",net_type*"_$(expr)_"*make_label_from_keys(inputs)*"_solver.jld")
        info("# writing to $out_fname")
        jldopen(out_fname, "w") do file
            write(file, "problem_dict", d)
            write(file, "global_rwd_range", g)
        end
    end
end
