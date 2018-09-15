using MCTS
include("roadnet_pursuer_generator_MDP.jl")
using ProgressMeter
#  using PyPlot
using JLD
using MicroLogging
using Base.Markdown
using DataFrames, CSV
include("road_net_visualize.jl")
include("utilities.jl")
include("self_confidence.jl")

rwd_global = []

####### NEED TO RE-VISIT HOW THESE FILES WORK, THERE ARE A LOT OF MULTIPLE IMPORTS
####### I PROBABLY NEED TO PACKAGE THIS STUFF TO AVOID ISSUES WHEN I TRY TO IMPORT @EVERYWHERE
####### THIS IS CAUSING ISSUES WHEN TRYING TO RUN IN PARALLEL, I GOT A "method too new to be called from this world context" error

function get_reward_distribution(g::MetaGraph, mdp::roadnet_with_pursuer; max_steps::Int64=25,
                        its_vals::Array{Int64}=[5],d_vals::Array{Int64}=[2],exp_vals::Array{Float64}=[5.],
                        num_repeats::Int64=10,discounted_rwd::Bool=true,
                        initial_state::roadnet_pursuer_state=roadnet_pursuer_state(1,4))
    rnet = mdp.road_net.gprops[:net_stats]
    net_diam = rnet.diam

    if discounted_rwd
        dis_str = "Discounted Rwd"
    else
        dis_str = "Rwd"
    end

    @info "calculating distribution"
    @debug "Using:" max_steps its_vals d_vals num_repeats rnet mdp.tprob mdp.discount mdp.road_net.gprops[:reward_dict] mdp.road_net.gprops[:exit_nodes]

    # TODO: investigate what the exploration constant does, i.e. does it scale with N?
    # This instantiates empty MCTS policies because MCTS runs "online"
    policy_tilde = Vector{MCTS.MCTSPlanner}(0)
    hist_tilde = Vector{HistoryRecorder}(0)
    its_axis = Vector{Int64}(0)
    d_axis = Vector{Int64}(0)

    for i in 1:length(its_vals)
        for d in d_vals
            its = its_vals[i]
            s = MCTSSolver(n_iterations=its,depth=d,exploration_constant=exp_vals[i],enable_tree_vis=true)
            push!(its_axis,its)
            push!(d_axis,d)
            push!(policy_tilde,solve(s,mdp))
            push!(hist_tilde,HistoryRecorder(max_steps=max_steps))
        end
    end
    @debug "Total number of solvers being tested:" length(policy_tilde)

    function reward_grab(mdp::roadnet_with_pursuer,policy::MCTS.MCTSPlanner,hist::HistoryRecorder,s::roadnet_pursuer_state;discounted::Bool=true,return_hist::Bool=false)
        hist = simulate(hist,mdp,policy,s)
        if return_hist
            return hist
        end

        # discounted reward takes time to solution into account
        if discounted
            return discounted_reward(hist)
        else
            return undiscounted_reward(hist)
        end
    end

    PT = repmat(policy_tilde,1,num_repeats)
    HT = repmat(hist_tilde,1,num_repeats)
    IA = repmat(its_axis,1,num_repeats)
    DA = repmat(d_axis,1,num_repeats)
    @debug "PT $(size(PT)),HT $(size(HT)),IA $(size(IA)),DA $(size(DA))"
    rewards = Array{Float64}(size(PT))
    sim_time = Array{Int64}(size(PT))
    idx = 1

    @info md"# Simulations"
    tic()

    if nprocs() > 1
        q = []
        for pol in PT
            value = Sim(mdp,pol,initial_state,max_steps=max_steps)
            push!(q,value)
        end
        sim_results = run_parallel(q) do sim,hist
            display(discounted_reward(hist))
            return [:steps=>n_steps(hist), :reward=>discounted_reward(hist)]
        end

        rewards = [float(x) for x in sim_results[:reward]] #need to convert to float, because sim_results[:reward] is a union of floats and missing values, even though there aren't any missing values...

    else
        pm = Progress(length(PT[:]),1)
        for (p,h) in zip(PT,HT)
            @info "Running Simulations" progress=idx/length(PT)
            rewards[idx] = reward_grab(mdp,p,h,initial_state,discounted=discounted_rwd)
            hist = reward_grab(mdp,p,h,initial_state,discounted=discounted_rwd,return_hist=true)
            sim_time[idx] = length(hist)
            if false
              @debug print_hist(mdp,hist)
            end
            next!(pm)
            idx += 1
        end
    end
    @info "Completed $(length(PT)) Simulations in $(toq()) seconds"

    R = reshape(rewards,length(its_axis),:)
    ST = reshape(sim_time,length(its_axis),:)

    # Save off data
    jld_fname = "data/$(num_repeats)_$(DateTime(now())).jld"
    @info "saving data to `$jld_fname`"
    JLD.save(jld_fname,"its_axis",its_axis,"d_axis",d_axis,"rewards",rewards,
         "u_vals",mean(rewards,2),"max_steps",max_steps,"R",R,"ST",ST,"IA",IA,"DA",DA)

    return vec(R) #convert to vector
end

function netprops2array(g::roadnet_with_pursuer)
    discount = g.discount
    exits = g.exit_nodes
    rwd_states = g.reward_states # not used right now
    rwd_vals = g.reward_vals # not used right now
    tprob = g.tprob

    num_exit_nodes = length(exits)

    rnet = g.road_net
    rnet_props = rnet.gprops
    exit_rwd = rnet_props[:reward_dict][:exit]
    caught_rwd = rnet_props[:reward_dict][:caught]
    sensor_rwd = rnet_props[:reward_dict][:sensor]
    rnet_stats = rnet_props[:net_stats]
    avg_degree = rnet_stats.average_degree
    deg_variance = rnet_stats.degree_variance
    diam = rnet_stats.diam
    max_degree = rnet_stats.maximum_degree
    N = rnet_stats.nv
    E = rnet_stats.ne

    netprops_ary = [discount tprob num_exit_nodes exit_rwd caught_rwd sensor_rwd avg_degree deg_variance diam max_degree N E]

    return netprops_ary

end

function make_training_data(net::Dict)
    net_dict = net[:sims]
    reps = net_dict[:repeats]
    log_to_file = net_dict[:log_to_file]
    log_fname = net_dict[:log_fname]
    log_lvl = net_dict[:log_lvl]
    sim_steps = net_dict[:steps]
    dis_rwd = net_dict[:dis_rwd]
    fname = net[:fname]

    fn = joinpath(fldr,fname)

    make_training_data(data_fname=fn,repeats=reps,sim_steps=sim_steps,dis_rwd=dis_rwd)
end

function make_training_data(;data_fname::String="nets",logtofile::Bool=false, logfname::String="logs/$(now()).log",loglvl::Symbol=:debug,repeats::Int64=25,sim_steps::Int64=-1,dis_rwd::Bool=false)

    # initialize the DataFrame
    training_data = DataFrame(graphID=Int64[],discount=Float64[],tprob=Float64[],num_exit_nodes=Float64[],exit_rwd=Float64[],
                              caught_rwd=Float64[],sensor_rwd=Float64[],avg_degree=Float64[],deg_variance=Float64[],
                              diam=Float64[],max_degree=Float64[],N=Float64[],E=Float64[],its=Float64[],
                              e_mcts=Float64[],d_mcts=Float64[],steps=Float64[],repeats=Float64[],
                              exit_distance=Float64[],pursuer_distance=Float64[],X3_1=Float64[],X3_2=Float64[],X4=Float64[],
                              mean=Float64[],median=Float64[],moment_2=Float64[],moment_3=Float64[],
                              moment_4=Float64[],moment_5=Float64[],moment_6=Float64[],moment_7=Float64[],
                              moment_8=Float64[],moment_9=Float64[],moment_10=Float64[])
    #### Logging
    if logtofile
        buffer = IOBuffer()
        logger = SimpleLogger(buffer)
        configure_logging(min_level=loglvl)
    else
        logger = global_logger()
        configure_logging(min_level=loglvl)
    end

    # get list of graphs
    problem_dict = jldopen("$data_fname.jld", "r") do file
        @info "Grabbing Pre-Generated Networks"
        read(file,"problem_dict")
    end

    global_rwd_range = Array{Float64}(2,1) # idx 1 is minimun, idx 2 is maximum

    i = 0
    for problem in problem_dict
        i += 1
        display(problem_dict)
        println()
        display(problem)
        println()
        net_num = problem[1]
        problem = problem[2]
        if :error in keys(problem)
            println(problem)
            continue
        else
            @info md"# Processing Network $i of $(length(problem_dict))"
            g = problem[:graph]
            t = problem[:trans_prob]
            d = problem[:discount]

            # do stuff
            # i'm thinking using the same network multiple times using different discount factors, and transition probabilities, this will do two things: 1) make training data better, 2) increase total amount of training data
            mdp = roadnet_with_pursuer(g,tp=t,d=d)

            its = [problem[:mcts_its]] # perhaps base this on transition probability
            d_mcts = [problem[:mcts_depth]]
            e_mcts = [problem[:mcts_e]]

            if sim_steps == -1
                # -1 signifies auto calculation
                # throw the book at it!
                steps = 20*mdp.road_net.gprops[:net_stats].diam
            else
                steps = sim_steps
            end

            start = problem[:evader_start]
            escape = problem[:exit_loc]
            pursuer = problem[:pursuer_start]
            start_state = roadnet_pursuer_state(start,pursuer)
            exit_distance = a_star(mdp.road_net,start,escape)
            pursuer_distance = a_star(mdp.road_net,start,pursuer)

            r_dist = []
            with_logger(logger) do
                r_dist = get_reward_distribution(g,mdp,its_vals=its,d_vals=d_mcts,exp_vals=e_mcts,max_steps=steps,num_repeats=repeats,discounted_rwd=dis_rwd,initial_state=start_state)
            end

            # update global reward range
            minr = minimum(r_dist)
            maxr = maximum(r_dist)
            if i==1
                global_rwd_range[1] = minr
                global_rwd_range[2] = maxr
            else
                minr < global_rwd_range[1] && (global_rwd_range[1] = minr)
                maxr > global_rwd_range[2] && (global_rwd_range[2] = maxr)
            end

            @info "calculating X3 data"
            SQ_data = X3(r_dist)
            @debug SQ_data

            @info "calculating X4 data"
            OA_data = X4(r_dist,threshold=mean(r_dist))
            @debug OA_data

            @info "calculating statistical moments"
            r_dist_moments = [mean(r_dist) median(r_dist) moment(r_dist,2) moment(r_dist,3) moment(r_dist,4) moment(r_dist,5) moment(r_dist,6) moment(r_dist,7) moment(r_dist,8) moment(r_dist,9) moment(r_dist,10)]
            @debug r_dist_moments

            @info "collecting net props"
            mdp_props_ary = netprops2array(mdp)
            @debug mdp_props_ary

            @info "collecting solver props"
            solver_props_ary = [its[1] e_mcts[1] d_mcts[1] steps repeats length(exit_distance) length(pursuer_distance)]
            @debug solver_props_ary

            @info "combining to a row"
            data_entry = [net_num mdp_props_ary solver_props_ary SQ_data OA_data r_dist_moments]
            @debug length(net_num) length(mdp_props_ary) length(solver_props_ary) length(SQ_data) length(r_dist_moments)
            @debug data_entry
            @debug length(data_entry)
            @debug training_data
            push!(training_data,data_entry)

            # store training data with individual network for future use
            training_data_dict = Dict()
            n_cntr = 1
            for n in names(training_data)
                training_data_dict[n] =  data_entry[n_cntr]
                n_cntr += 1
            end
            problem[:training_data] = training_data_dict
            problem[:training_data][:r_dist] = r_dist
        end
    end
    CSV.write("$data_fname.csv",training_data)

    # re-write original file with "r_dist" object added to each network
    jldopen("$data_fname.jld", "w") do file
        write(file, "problem_dict", problem_dict)
        write(file, "global_rwd_range", global_rwd_range)
    end

    if logtofile
        # using the "open(...) do f ... syntax printed the length of the file at the end, this ways doesn't do that
        f = open(logfname,"w")
        logtxt = strip(String(take!(buffer)));
        logtxt = replace(logtxt, r"^\e\[1m\e\[..m(.- )\e\[39m\e\[22m", s"\1")
        write(f,logtxt)
        close(f)
    end
end
