using MCTS
using Plots, PlotRecipes
using ProgressMeter
using JLD
using Memento # logging utility
include("roadnet_pursuer_generator_MDP.jl")
include("outcome_assessment.jl")
include("road_net_visualize.jl")
include("utilities.jl")


function run_experiment(g::MetaGraph, mdp::roadnet_with_pursuer, logger::Logger; max_steps::Int64=25,
                        its_vals::Array{Int64}=[5],d_vals::Array{Int64}=[2],num_repeats::Int64=10)
    info(logger,"$(now()) -- Starting new experiment")
    info(logger,"running with $g, $mdp, max_steps $max_steps, its_vals $its_vals, d_vals $d_vals, and num_repeats $num_repeats")
    rnet = mdp.road_net.gprops[:net_stats]
    net_diam = rnet.diam

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
    p_meter = Progress(length(PT[:]),desc="Calculating Utilities:",color=:yellow,barglyphs=BarGlyphs("[=> ]"))

    for (p,h) in zip(PT,HT)
        utilities[idx] = reward_grab(mdp,p,h,starting_state)
        ProgressMeter.next!(p_meter)
        idx += 1
    end

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
    violin(D2[:,2],D2[:,3])
    #corrplot(D)
    #  corrplot(D2, labels=["its","d","U"])
    #  scatter3d(its_axis,d_axis,mean(utilities,2)[:])
end

function main()
    g = original_roadnet(exit_rwd=1000.,sensor_rwd=-50.)
    mdp = roadnet_with_pursuer(g,tp=0.5)

    logger = Memento.config("info";fmt="[{level} | {name}]: {msg}")
    add_handler(logger,DefaultHandler("logs/$(now()).log"),"file-logging")

    its_rng = (1., 10000.)
    d_rng = (1, 2*mdp.road_net.gprops[:net_stats].diam)
    #  its_vals = Int.(round.(latin_hypercube_sampling([its_rng[1]],[its_rng[2]],25)))
    #  d_vals = Int.(round.(latin_hypercube_sampling([d_rng[1]],[d_rng[2]],10)))

    run_experiment(g,mdp,logger)
end
