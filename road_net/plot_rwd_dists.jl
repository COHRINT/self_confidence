function plot_rwd_dists(net_num,xQ,xP,c_data,t_dist,o;fldr::String="figs",fname::String="",outcome::Symbol=:fail)
    cm = mean(c_data)
    cs = std(c_data)

    tm = t_dist[:X3_1][1]
    ts = t_dist[:X3_2][1]

    pygui(false)
    f,ax = subplots(1,2)
    f[:set_size_inches](8,4)

    ax[1][:hist](c_data)
    ax[1][:set_title](@sprintf("Candidate Dist. Drew: %0.2f",o))
    mc = @sprintf("Mean: %0.2f, Var: %0.2f",cm,cs)
    ax[1][:axvline](cm,color=:red,label=mc)
    ax[1][:legend]()

    ax[2][:hist](rand(Normal(tm,ts),1000))
    mt = @sprintf("Mean: %0.2f, Var: %0.2f",tm,ts)
    ax[2][:axvline](tm,color=:red,label=mt)
    ax[2][:set_title]("Trusted Distribution")
    ax[2][:legend]()

    f[:suptitle](@sprintf("Network: %d---Outcome %s---xQ: %0.2f, xP: %0.2f",net_num,string(outcome),xQ,xP))
    #  f[:tight_layout]()
    if fname == ""
        fname = "net_num$net_num"
    end
    file_name = joinpath(fldr,fname)
    println("saving to: $file_name")
    PyPlot.savefig("$file_name.pdf",dpi=300,transparent=true)
end
