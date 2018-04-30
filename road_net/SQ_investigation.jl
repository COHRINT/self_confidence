@everywhere using PyPlot
@everywhere using Roadnet_MDP
@everywhere using DataFrames, Query
@everywhere using POMDPToolbox
@everywhere using MCTS
@everywhere include("roadnet_pursuer_generator_MDP.jl")
@everywhere using ProgressMeter
@everywhere using JLD
@everywhere using MicroLogging
@everywhere using Base.Markdown
@everywhere include("road_net_visualize.jl")
@everywhere include("utilities.jl")
@everywhere include("self_confidence.jl")
@everywhere include("send_mail.jl")

function run_experiment(g::MetaGraph, mdp::roadnet_with_pursuer; max_steps::Int64=25,
                        its_vals::Array{Int64}=[5],d_vals::Array{Int64}=[2],exp_vals::Array{Float64}=[5.],
                        num_repeats::Int64=10,discounted_rwd::Bool=true,data_fname::String="data/test")
    rnet = mdp.road_net.gprops[:net_stats]
    net_diam = rnet.diam

    if discounted_rwd
        dis_str = "Discounted Rwd"
    else
        dis_str = "Rwd"
    end

    @info md"# Beginning Experiment"
    @debug "Using:" max_steps its_vals d_vals num_repeats rnet mdp.tprob mdp.discount mdp.reward_vals mdp.reward_states

    # TODO: investigate what the exploration constant does, i.e. does it scale with N?
    # This instantiates empty MCTS policies because MCTS runs "online"
    policy_tilde = Vector{MCTS.MCTSPlanner}(0)
    hist_tilde = Vector{HistoryRecorder}(0)
    its_axis = Vector{Int64}(0)
    d_axis = Vector{Int64}(0)

    for i in 1:length(its_vals)
        for d in d_vals
            its = its_vals[i]
            #  s = MCTSSolver(n_iterations=its,depth=d,exploration_constant=exp_vals[i],enable_tree_vis=true)
            s = MCTSSolver(n_iterations=its,depth=d,exploration_constant=exp_vals[i],enable_tree_vis=false)
            push!(its_axis,its)
            push!(d_axis,d)
            push!(policy_tilde,solve(s,mdp))
            push!(hist_tilde,HistoryRecorder(max_steps=max_steps))
        end
    end
    @debug "Total number of solvers being tested:" length(policy_tilde)

    initial_state = roadnet_pursuer_state(1,4)

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
    R = zeros(size(DA))

    @info md"# Simulations"
    tic()
    if nprocs() > 1
        q = []
        for pol in PT
            value = Sim(mdp,pol,initial_state,max_steps=max_steps,metadata=Dict(:d=>pol.solver.depth))
            push!(q,value)
        end
        sim_results = run_parallel(q) do sim,hist
            if discounted_rwd
                return [:steps=>n_steps(hist), :reward=>discounted_reward(hist)]
            else
                return [:steps=>n_steps(hist), :reward=>undiscounted_reward(hist)]
            end
        end
        #  println(sim_results)

        for k = 1:length(d_vals)
            i = d_vals[k]
            di = @from r in sim_results begin
                 @where r.d == i
                 @select {r.reward}
                 @collect DataFrame
             end
            R[k,:] = [float(x) for x in di[:reward]] #need to convert to float, because sim_results[:reward] is a union of floats and missing values, even though there aren't any missing values...
        end
    else
        pm = Progress(length(PT[:]),1)
        for (p,h) in zip(PT,HT)
            @info "Running Simulations" progress=idx/length(PT)
            rewards[idx] = reward_grab(mdp,p,h,initial_state,discounted=discounted_rwd)
            hist = reward_grab(mdp,p,h,initial_state,discounted=discounted_rwd,return_hist=true)
            sim_time[idx] = length(hist)
            if false
              @info print_hist(mdp,hist)
            end
            next!(pm)
            idx += 1
        end
        R = reshape(rewards,length(its_axis),:)
    end
    @info "Completed $(length(PT)) Simulations in $(toq()) seconds"

    ST = reshape(sim_time,length(its_axis),:)

    # Save off data
    #  @save "data/$(num_repeats)_$(DateTime(now())).jld2" its_axis d_axis utilities max_steps U ST IA DA
    # @save "data/$(num_repeats)_$(DateTime(now())).jld2" its_axis d_axis utilities mean(utilities,2)[:] max_steps U ST IA DA
    #  JLD2.save("data/$(num_repeats)_$(DateTime(now())).jld2",Dict("its_axis" => its_axis,
             #  "d_axis" => d_axis,"utilities" => utilities,"u_vals" => mean(utilities,2)[:],
             #  "max_steps" => max_steps,"R" => R, "ST" => ST,"IA" => IA,"DA" => DA))
     JLD.save("$data_fname.jld","its_axis",its_axis,"d_axis",d_axis,"rewards",rewards,
         "u_vals",mean(rewards,2),"max_steps",max_steps,"R",R,"ST",ST,"IA",IA,"DA",DA)

    ####### Plotting #######
    ## Reward vs d
    #  D = [its_axis d_axis mean(R,2)]
    println("!!!!!!!!!!!!!!!!!!")
    println("min R: $(minimum(R)), max R: $(maximum(R))")
    println("!!!!!!!!!!!!!!!!!!")

    D2 = [IA[:] DA[:] R[:]]
    if length(unique(R[:])) == 1
        # plots will fail, because there aren't enough unique y's
        @warn "not enough unique y's, exiting without producing plots"
        return
    end

    #  @debug "IA $IA\nDA $DA\nR $R"
    #  @debug "D2 $D2"
    #  @debug "$(D2[:,2]), $(D2[:,3])"

    if false
        fig,ax_ary = PyPlot.subplots(2,1,sharex=true)
        fig[:set_size_inches](8.0,8.0)

        (x_vals,x_ticks,x_lbls,x_rng) = return_indices(DA[:])

        vline_frac = (mdp.road_net.gprops[:net_stats].diam - x_rng[1])/(x_rng[2]-x_rng[1])
        vline_tick = x_ticks+vline_frac*(maximum(x_ticks)-minimum(x_ticks))

        @debug typeof(x_vals) size(x_vals)
        @debug typeof(x_ticks) size(x_ticks) x_ticks
        @debug typeof(R) size(R)
        # points = # of points where the Kernel Density Estimator is estimated
        # bw_method = bandwidth of the KDE
        ax_ary[1][:violinplot](R',x_ticks,widths=0.5,points=100,showmedians=true)
        ax_ary[1][:axvline](x=vline_tick,color="red",lw=1)

        ax_ary[2][:violinplot](ST',x_ticks,widths=0.5,points=100,showmedians=true)
        ax_ary[2][:axvline](x=vline_tick,color="red",lw=1,label="Net Diam.")

        ax_ary[1][:set_ylabel]("$dis_str")
        ax_ary[1][:xaxis][:set_ticklabels]([])
        ttl_str = "$dis_str vs MCTS depth\nTrans_prob = $(mdp.tprob), MCTS Parameters: N = $its_vals, e=$exp_vals\nD= $d_vals\nBased on $num_repeats separate simulations"
        ax_ary[1][:set_title](ttl_str)

        ax_ary[2][:set_xticks](x_ticks)
        ax_ary[2][:xaxis][:set_ticklabels](x_lbls)
        ax_ary[2][:set_xlabel]("MCTS Depth")
        ax_ary[2][:set_ylabel]("Time to termination")

        ax_ary[2][:legend](loc=1,bbox_to_anchor=(1.0,1.15))
    else
        fig,ax_ary = PyPlot.subplots(1,1,sharex=true)
        fig[:set_size_inches](8.0,4.0)

        (x_vals,x_ticks,x_lbls,x_rng) = return_indices(DA[:])

        vline_frac = (mdp.road_net.gprops[:net_stats].diam - x_rng[1])/(x_rng[2]-x_rng[1])
        vline_tick = minimum(x_ticks)+vline_frac*(maximum(x_ticks)-minimum(x_ticks))

        # points = # of points where the Kernel Density Estimator is estimated
        # bw_method = bandwidth of the KDE
        @info "shape of R is: $(size(R))"
        @info "x_ticks: $x_ticks"
        @info "network diameter = $(mdp.road_net.gprops[:net_stats].diam)"
        @info "vline tick_location= $vline_tick"
        trusted_solver_num = 9
        ax_ary[:violinplot](R[1:end .!= trusted_solver_num,:]',x_ticks[1:end .!= trusted_solver_num],widths=0.5,points=100,showmedians=true)
        trusted_violin = ax_ary[:violinplot](R[trusted_solver_num,:]',[x_ticks[trusted_solver_num]],widths=0.5,points=100,showmedians=true)
        #  @info trusted_violin["bodies"]
        #  for pc in trusted_violin["bodies"]
            #  pc.set_facecolor("red")
            #  pc.set_edgecolor("red")
        #  end

        if false
            #add vline for network diameter, interesting to see
            ax_ary[:axvline](x=vline_tick,color="red",lw=1,label="Net. Diam.")
        end
        for i in x_ticks
            #  display(R[i,:])
            SQ = X3(R[i,:],R[trusted_solver_num,:],global_rwd_range=[minimum(R),maximum(R)])
            println("SQ: $SQ")
            #  @info "X3 at $i: $SQ"
            ax_ary[:annotate]("SQ: $(@sprintf("%0.2f",SQ))",xy=(i,mean(R[i,:])),size=6)
        end

        ax_ary[:set_ylabel]("$dis_str")
        ax_ary[:set_xticks](x_ticks)
        ax_ary[:xaxis][:set_ticklabels](x_lbls)
        ax_ary[:set_xlabel]("MCTS Depth")
        #  ax_ary[:legend](loc=1,bbox_to_anchor=(1.0,1.15))
        ax_ary[:set_title]("Reward vs. MCTS Depth")
    end

    if current_logger().default_min_level == MicroLogging.Debug
        ttl_str = "$dis_str vs MCTS depth\nTrans_prob = $(mdp.tprob), MCTS Parameters: N = $its_vals, e=$exp_vals\nD= $d_vals\nBased on $num_repeats separate simulations"
        ax_ary[:set_title](ttl_str)
        fig[:tight_layout]()
        @info "displaying plot(s)"
        show()
    else
        ttl_str = "$dis_str vs MCTS depth\nTrans_prob = $(mdp.tprob), MCTS Parameters: N = $its_vals, e=$exp_vals\nD= $d_vals\nBased on $num_repeats separate simulations"
        ax_ary[:set_title](ttl_str)
        fig[:tight_layout]()
        @info "saving plot(s) to file"
        @info "filename: $data_fname"
        PyPlot.savefig("$data_fname.pdf",dpi=300,transparent=true)
    end
end

function main(;logtofile::Bool=false, logfname::String="logs/$(now()).log",loglvl::Symbol=:debug,data_fname="",its_rng::Array{Int64}=[100],e_vals::Array{Float64}=[5.],send_mail_when_done::Bool=true)
    #### Logging
    if logtofile
        buffer = IOBuffer()
        logger = SimpleLogger(buffer)
        configure_logging(min_level=loglvl)
    else
        logger = global_logger()
        configure_logging(min_level=loglvl)
    end

    ext_rwd = 2000.
    cgt_rwd = -2000.

    g = original_roadnet(exit_rwd=ext_rwd,caught_rwd=cgt_rwd,sensor_rwd=-200.)
    mdp = roadnet_with_pursuer(g,tp=0.7,d=0.9)

    d_rng = collect(1:10)
    steps = 100 # number of steps the simulation runs
    repeats = 250 # how many times to repeat each simlation
    dis_rwd = false

    if data_fname == ""
        data_fname = "data/sq_roadnet_mcts_i$(its_rng[1])e$(Int(e_vals[1]))"
    end

    with_logger(logger) do
        run_experiment(g,mdp,its_vals=its_rng,d_vals=d_rng,exp_vals=e_vals,max_steps=steps,
                       num_repeats=repeats,discounted_rwd=dis_rwd,data_fname=data_fname)
    end

    if logtofile
        # using the "open(...) do f ... syntax printed the length of the file at the end, this ways doesn't do that
        f = open(logfname,"w")
        logtxt = strip(String(take!(buffer)));
        logtxt = replace(logtxt, r"^\e\[1m\e\[..m(.- )\e\[39m\e\[22m", s"\1")
        write(f,logtxt)
        close(f)
    end
    if on_gcloud() && send_mail_when_done
        #if we're on Google cloud, then send an email when code is done
        hname = on_gcloud(return_name=true)
        body = "The code on $hname has finished running"
        subject = "Done running code on $hname"
        send_mail(subject,body)
    end
end

function main2(;logtofile::Bool=false, logfname::String="logs/$(now()).log",loglvl::Symbol=:debug,data_fname="",its_rng::Array{Int64}=[1000],e_vals::Array{Float64}=[5.],send_mail_when_done::Bool=true)
    #### Logging
    if logtofile
        buffer = IOBuffer()
        logger = SimpleLogger(buffer)
        configure_logging(min_level=loglvl)
    else
        logger = global_logger()
        configure_logging(min_level=loglvl)
    end

    ext_rwd = 2000.
    cgt_rwd = -2000.

    g = medium_roadnet(exit_rwd=ext_rwd,caught_rwd=cgt_rwd,sensor_rwd=-200.)
    mdp = roadnet_with_pursuer(g,tp=0.7,d=0.95)

    d_rng = collect(1:3:30)
    steps = 100 # number of steps the simulation runs
    repeats = 250 # how many times to repeat each simlation
    dis_rwd = false

    if data_fname == ""
        data_fname = "data/sq_mednet_mcts_i$(its_rng[1])e$(Int(e_vals[1]))"
    end

    with_logger(logger) do
        run_experiment(g,mdp,its_vals=its_rng,d_vals=d_rng,exp_vals=e_vals,max_steps=steps,
                       num_repeats=repeats,discounted_rwd=dis_rwd,data_fname=data_fname)
    end

    if logtofile
        # using the "open(...) do f ... syntax printed the length of the file at the end, this ways doesn't do that
        f = open(logfname,"w")
        logtxt = strip(String(take!(buffer)));
        logtxt = replace(logtxt, r"^\e\[1m\e\[..m(.- )\e\[39m\e\[22m", s"\1")
        write(f,logtxt)
        close(f)
    end

    if on_gcloud() && send_mail_when_done
        #if we're on Google cloud, then send an email when code is done
        hname = on_gcloud(return_name=true)
        body = "The code on $hname has finished running"
        subject = "Done running code on $hname"
        send_mail(subject,body)
    end
end
