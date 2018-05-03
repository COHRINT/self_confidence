using JLD
using PyPlot
include("utilities.jl")
include("self_confidence.jl")
pygui(false)

searchdir(path,key) = filter(x->contains(x,key), readdir(path))

folder = "gcloud_data"

for file in searchdir(folder,"jld")
    println("Processing: $file")
    fname = split("$(folder)/$file",".")[1]
    fsize = 12

    data = JLD.load("$fname.jld")
    max_steps = data["max_steps"]
    u_vals = data["u_vals"]
    its_axis = data["its_axis"]
    d_axis = data["d_axis"]
    DA = data["DA"]
    R = data["R"]
    rewards = data["rewards"]
    IA = data["IA"]
    ST = data["ST"]

    fig,ax_ary = PyPlot.subplots(1,1,sharex=true)
    fig[:set_size_inches](8.0,4.0)

    (x_vals,x_ticks,x_lbls,x_rng) = return_indices(DA[:])

    #  vline_frac = (mdp.road_net.gprops[:net_stats].diam - x_rng[1])/(x_rng[2]-x_rng[1])
    #  vline_tick = minimum(x_ticks)+vline_frac*(maximum(x_ticks)-minimum(x_ticks))

    # points = # of points where the Kernel Density Estimator is estimated
    # bw_method = bandwidth of the KDE
    trusted_solver_num = 9
    #  ax_ary[:violinplot](R[1:end .!= trusted_solver_num,:]',x_ticks[1:end .!= trusted_solver_num],widths=0.5,points=250,showmeans=false,showmedians=true,showextrema=true)
    tbp = ax_ary[:boxplot](R[trusted_solver_num,:]',positions=[x_ticks[trusted_solver_num]],widths=0.25,notch=false,showfliers=false,autorange=true,patch_artist=true)
    cbp = ax_ary[:boxplot](R[1:end .!= trusted_solver_num,:]',positions=x_ticks[1:end .!= trusted_solver_num],widths=0.25,notch=false,showfliers=false,autorange=true)
    #  for x in x_ticks
        #  ax_ary[:scatter](float(x)*ones(length(R[x,:])),R[x,:],alpha=0.1)
    #  end
    #  trusted_violin = ax_ary[:violinplot](R[trusted_solver_num,:]',[x_ticks[trusted_solver_num]],widths=0.5,points=250,showmeans=false,showmedians=true,showextrema=true)
    for element in ["boxes","whiskers","means","caps"]
        PyPlot.setp(cbp[element],color=:red)
    end
    for element in ["boxes","whiskers","means","caps"]
        PyPlot.setp(tbp[element],color=:blue)
    end
    for patch in tbp["boxes"]
        patch[:set_facecolor](:blue)
        patch[:set_alpha](0.35)
    end

    for i in x_ticks
        #  display(R[i,:])
        SQ = X3(R[i,:],R[trusted_solver_num,:],global_rwd_range=[minimum(R),maximum(R)])
        println("SQ: $SQ")
        #  @info "X3 at $i: $SQ"
        ax_ary[:annotate]("SQ: $(@sprintf("%0.2f",SQ))",xy=(i,mean(R[i,:])),size=8)
    end

    ax_ary[:set_ylabel]("Reward",size=fsize)
    ax_ary[:set_xticks](x_ticks)
    ax_ary[:xaxis][:set_ticklabels](x_lbls)
    ax_ary[:set_xlabel]("MCTS Depth",size=fsize)
    ax_ary[:set_title]("Reward vs. MCTS Depth",size=fsize)

    PyPlot.savefig("$fname.pdf",dpi=300,transparent=true)
end
