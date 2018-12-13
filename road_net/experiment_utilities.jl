function make_useful_fname(exp_dict::Dict,cond_key::Int64)
    exp_name = exp_dict[:name]
    cond_role = exp_dict[:conditions][cond_key][:role]

    fname = exp_name * "_" * string(cond_role) * "_condition_" * string(cond_key)
end
function make_condition_dict(;num_nets::Int64=2,t_bnds::Array{Float64}=[0.0,1.0],
                       e_bnds::Array{Float64}=[1000.],mcts_d::Array{Int64}=[2],
                       deg_bnds::Array{Float64}=[4.],net_type::Symbol=:random,d_bnds::Array{Float64}=[0.95],
                       exit_rwd_bnds::Array{Float64}=[2000.],sense_rwd_bnds::Array{Float64}=[-100.],
                       caught_rwd_bnds::Array{Float64}=[-2000.],mcts_its_bnds::Array{Int64}=[1000],
                       seed::Int64=5,n_bnds::Array{Int64}=[13],repeats::Int64=250,steps::Int64=-1,
                       dis_rwd::Bool=false,log_to_file::Bool=false,role::Symbol=:candidate)
    # this is a default condition, to reduce the need for typing a bunch of repeated information
    # num_nets -- number of networks to produce
    # t_bnds -- bounds for transition probabilities of produced networks,
    #           single parameter indicates that the parameter will be constant across all networks
    # e_bnds -- bounds for mcts exploration constant of produced networks
    # mcts_d -- bounds for mcts depth of produced networks
    # deg_bnds -- bounds for degree of produced networks
    # net_type -- type of network produced -- :random or :original
    # d_bnds -- bound for discount factor of produced networks
    # exit_rwd_bnds -- bound for exit reward of produced networks
    # sense_rwd_bnds -- bound for sense reward of produced networks
    # caught_rwd_bnds -- bound for caught reward of produced networks
    # mcts_its_gnds -- bound for number of mcts iterations of produced networks
    # seed -- seed for random number generator
    # n_bnds -- bound for number of nodes in network, original has 13
    # repeats -- number of simulations that will be run to determine reward distribution
    # steps -- number of steps used in simulation, before simulation terminates
    # dis_rwd -- whether to use discounted reward or not (other possibility is un-discounted reward)
    # log_to_file -- whether to log to a file
    # role -- what role the condition will play in the overall experiment. :trusted, or :candidate

    net_dict = Dict(:num_nets=>num_nets,:t_bnds=>t_bnds,:e_bnds=>e_bnds,:solver_depth=>mcts_d,
                    :deg_bnds=>deg_bnds,:net_type=>net_type,:n_bnds=>n_bnds,:d_bnds=>d_bnds,
                    :exit_rwd_bnds=>exit_rwd_bnds,:sense_rwd_bnds=>sense_rwd_bnds,
                    :caught_rwd_bnds=>caught_rwd_bnds,:mcts_its_bnds=>mcts_its_bnds,:seed=>seed)
    sims_dict = Dict(:repeats=>repeats,:steps=>steps,:dis_rwd=>dis_rwd,:log_to_file=>log_to_file,
                      :log_fname=>"logs/$(now())",:log_lvl=>:debug)
    cond = Dict(:role=>role,:nets=>net_dict,:sims=>sims_dict)

    return cond
end
if experiment_name == "transition_vary"
    include("experiment_params/transition_vary.jl")
elseif experiment_name == "mturk_fast"
    include("experiment_params/mturk_fast.jl")
elseif experiment_name == "mturk"
    include("experiment_params/mturk.jl")
elseif experiment_name == "mturk_supplement"
    include("experiment_params/mturk_supplement.jl")
elseif experiment_name == "x4_test"
    include("experiment_params/x4_test.jl")
elseif experiment_name == "n_vary"
    include("experiment_params/n_vary.jl")
elseif experiment_name == "sense_vary"
    include("experiment_params/sense_vary.jl")
elseif experiment_name == "transition_e_vary"
    include("experiment_params/transition_e_vary.jl")
elseif experiment_name == "net_transition_vary"
    include("experiment_params/net_transition_vary.jl")
elseif experiment_name == "net_transition_sense_vary"
    include("experiment_params/net_transition_sense_vary.jl")
elseif experiment_name == "something else"
    #keeping empty for now
end

