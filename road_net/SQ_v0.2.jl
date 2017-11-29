using MCTS
using Plots, PlotRecipes
using ProgressMeter
using JLD
using MicroLogging
using Base.Markdown
include("roadnet_pursuer_generator_MDP.jl")
include("outcome_assessment.jl")
include("road_net_visualize.jl")
include("utilities.jl")

function run_experiment(g::MetaGraph, mdp::roadnet_with_pursuer; max_steps::Int64=25,
                        its_vals::Array{Int64}=[5],d_vals::Array{Int64}=[2],num_repeats::Int64=10)
    rnet = mdp.road_net.gprops[:net_stats]
    net_diam = rnet.diam

    @info md"# Beginning Experiment"
    @debug "Using:" max_steps its_vals d_vals num_repeats rnet mdp.tprob mdp.discount mdp.reward_vals mdp.reward_states

    # TODO: investigate what the exploration constant does, i.e. does it scale with N?
    # This instantiates empty MCTS policies because MCTS runs "online"
    policy_tilde = Vector{MCTS.MCTSPlanner}(0)
    hist_tilde = Vector{HistoryRecorder}(0)
    its_axis = Vector{Int64}(0)
    d_axis = Vector{Int64}(0)

    for its in its_vals
        for d in d_vals
            s = MCTSSolver(n_iterations=its,depth=d,exploration_constant=5.,enable_tree_vis=true)
            push!(its_axis,its)
            push!(d_axis,d)
            push!(policy_tilde,solve(s,mdp))
            push!(hist_tilde,HistoryRecorder(max_steps=max_steps))
        end
    end
    @debug "Total number of solvers being tested:" length(policy_tilde)

    starting_state = roadnet_pursuer_state(1,4)

    function reward_grab(mdp::roadnet_with_pursuer,policy::MCTS.MCTSPlanner,hist::HistoryRecorder,s::roadnet_pursuer_state)
        hist = simulate(hist,mdp,policy,s)

        # discounted reward takes time to solution into account
        return discounted_reward(hist)
    end

    PT = repmat(policy_tilde,1,num_repeats)
    HT = repmat(hist_tilde,1,num_repeats)
    IA = repmat(its_axis,1,num_repeats)
    DA = repmat(d_axis,1,num_repeats)
    utilities = Array{Float64}(size(PT))
    idx = 1
    #  p_meter = Progress(length(PT[:]),desc="Calculating Utilities:",color=:yellow,barglyphs=BarGlyphs("[=> ]"))

    @info md"# Simulations"
    tic()
    for (p,h) in zip(PT,HT)
        @info "Running Simulations" progress=idx/length(PT)
        utilities[idx] = reward_grab(mdp,p,h,starting_state)
        #  ProgressMeter.next!(p_meter)
        idx += 1
    end
    @info "Completed $(length(PT)) Simulations in $(toq()) seconds"

    #  U = reshape(utilities,length(policy_tilde),length(hist_tilde));
    #
    U = reshape(utilities,length(its_axis),:);
    U2 = reshape(utilities,:,length(its_axis));

    #  cx = repmat(its_axis',length(d_axis),1)
    #  cy = repmat(d_axis,1,length(its_axis))

    # Save off data
    JLD.save("data/$(num_repeats)_$(DateTime(now())).jld","its_axis",its_axis,
             "d_axis",d_axis,"utilities",utilities,"u_vals",mean(utilities,2)[:],
            "max_steps",max_steps)

    #  println("its: $(size(its_axis)), d: $(size(d_axis)), u: $(size(utilities))")
    D = [its_axis d_axis mean(utilities,2)]
    D2 = [IA[:] DA[:] U[:]]
    v = violin(D2[:,2],D2[:,3])
    if current_logger().default_min_level == MicroLogging.Debug
        @debug "displaying plot(s)"
        display(v)
    else
        @info "saving plot(s) to file"
        savefig(v,"test.png")
    end
    #corrplot(D)
    #  corrplot(D2, labels=["its","d","U"])
    #  scatter3d(its_axis,d_axis,mean(utilities,2)[:])
end

function main(;logtofile::Bool=false, logfname::String="logs/$(now()).log",loglvl::Symbol=:debug)
    #### Logging
    if logtofile
        buffer = IOBuffer()
        logger = SimpleLogger(buffer)
        configure_logging(min_level=loglvl)
    else
        logger = global_logger()
        configure_logging(min_level=loglvl)
    end

    g = original_roadnet(exit_rwd=1000.,sensor_rwd=-50.)
    mdp = roadnet_with_pursuer(g,tp=0.3)

    #  its_rng = (1., 10000.)
    its_rng = [10]
    #  d_rng = (1, 2*mdp.road_net.gprops[:net_stats].diam)
    d_rng = collect(1:10)
    #  its_vals = Int.(round.(latin_hypercube_sampling([its_rng[1]],[its_rng[2]],25)))
    #  d_vals = Int.(round.(latin_hypercube_sampling([d_rng[1]],[d_rng[2]],10)))

    with_logger(logger) do
        run_experiment(g,mdp,its_vals=its_rng,d_vals=d_rng)
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
