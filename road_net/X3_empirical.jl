using StatsBase
using Distributions
using StatPlots,Plots
using LaTeXStrings
include("self_confidence.jl")

# https://github.com/joshday/AverageShiftedHistograms.jl
# might be useful at some point

function hist_diff(h1::Histogram,h2::Histogram)
    @assert length(h1.weights) == length(h2.weights)
    h_diff = h1.weights - h2.weights

    bin_widths = diff(h1.edges[1])
    return h_diff, h1.edges[1][1:end-1]+bin_widths/2, bin_widths[1]
end

function moment_around_reference(val::Array{Float64},loc::Array{Float64},r_star::Float64,
                            s_range::Float64;return_vec::Bool=false)

    r_mom = []
    for i = 1:length(val)
        push!(r_mom,val[i]*loc[i])
    end

    if return_vec
        return r_mom/s_range
    else
        return sum(r_mom/s_range)
    end
end

function residual_moment(c::Array{Float64},t::Array{Float64};
                         r_star::Float64=mean(t),solver_rwd_range::Float64=1.,return_vec::Bool=false,
                        return_hists::Bool=true)

    #  display("c_exp:$(mean(c)), t_exp: $(mean(t))")
    edg_max = maximum([maximum(c), maximum(t)])
    edg_min = minimum([minimum(c), minimum(t)])
    #  display("edg_max: $edg_max, edg_min: $edg_min")
    hist_edges = linspace(edg_min,edg_max,100)
    #  display(hist_edges)

    # based on distribution with widest range, use that as the edges for the second distribution
    c_norm = normalize(fit(Histogram,c,hist_edges,closed=:left),mode=:probability)
    t_norm = normalize(fit(Histogram,t,hist_edges,closed=:left),mode=:probability)
    display(c_norm)
    display("max weight: $(maximum(c_norm.weights))")

    # r for residual
    r_val, r_loc,r_width = hist_diff(c_norm,t_norm)
    r_mom = moment_around_reference(r_val,r_loc,r_star,solver_rwd_range,return_vec=return_vec)
    display("r_mom: $r_mom")
    if return_hists
        return r_val, r_loc, r_width, r_mom, c_norm, t_norm
    else
        return r_val, r_loc, r_width, r_mom
    end
end
function max_residual_moment(c::Array{Float64},t::Array{Float64};
                         r_star::Float64=mean(t),solver_rwd_range::Float64=1.,return_vec::Bool=false,
                        return_hists::Bool=true)

    #  display("c_exp:$(mean(c)), t_exp: $(mean(t))")
    edg_max = maximum([maximum(c), maximum(t)])
    edg_min = minimum([minimum(c), minimum(t)])
    #  display("edg_max: $edg_max, edg_min: $edg_min")
    hist_edges = linspace(edg_min,edg_max,100)
    #  display(hist_edges)

    # based on distribution with widest range, use that as the edges for the second distribution
    c_norm = normalize(fit(Histogram,c,hist_edges,closed=:left),mode=:probability)
    t_norm = normalize(fit(Histogram,t,hist_edges,closed=:left),mode=:probability)

    # r for residual
    r_val, r_loc,r_width = hist_diff(c_norm,t_norm)

    t1 = mean(c)*1/std(c)
    t2 = mean(t)*1/std(t)
    t3 = t1-t2
    t4 = t3/solver_rwd_range
    println("p1: $t1, p2: $t2, p3: $t3")
    return t4
end

function max_lik(c::Array{Float64},t::Array{Float64};
                         r_star::Float64=mean(t),solver_rwd_range::Float64=1.,return_vec::Bool=false,
                        return_hists::Bool=true)

    #  display("c_exp:$(mean(c)), t_exp: $(mean(t))")
    edg_max = maximum([maximum(c), maximum(t)])
    edg_min = minimum([minimum(c), minimum(t)])
    #  display("edg_max: $edg_max, edg_min: $edg_min")
    hist_edges = linspace(edg_min,edg_max,100)
    #  display(hist_edges)

    # based on distribution with widest range, use that as the edges for the second distribution
    c_norm = normalize(fit(Histogram,c,hist_edges,closed=:left),mode=:probability)
    t_norm = normalize(fit(Histogram,t,hist_edges,closed=:left),mode=:probability)

    ind_max_c = indmax(c_norm.weights)
    c_max = c_norm.weights[ind_max_c]
    c_max_loc = hist_edges[ind_max_c]

    ind_max_t = indmax(t_norm.weights)
    t_max = t_norm.weights[ind_max_t]
    t_max_loc = hist_edges[ind_max_t]

    # r for residual
    r_val, r_loc,r_width = hist_diff(c_norm,t_norm)
    r_mom = moment_around_reference(r_val,r_loc,r_star,solver_rwd_range,return_vec=return_vec)
    display("r_mom: $r_mom")
    if return_hists
        return r_val, r_loc, r_width, r_mom, c_norm, t_norm
    else
        return r_val, r_loc, r_width, r_mom
    end
end

function residual_moment(c::Float64,t::Float64;
                         r_star::Float64=mean(t),solver_rwd_range::Float64=1.,return_vec::Bool=false)

    r_mom = moment_around_reference([1. -1.],[c t],t,solver_rwd_range)
    if return_vec
        return r_mom/solver_rwd_range
    else
        return sum(r_mom/solver_rwd_range)
    end
end

function X3_emp_histogram()
    srand(12345)
    # t for trusted, c for candidate
    t1 = randn(500)*50
    c = t1
    c2 = (randn(1000)+10)*30
    c3 = [randn(1000)-10;randn(1000)+10]*30

    T = [t1,t1,t1]
    C = [c,c2,c3]
    solver_reward_range = [3000.,100.,100.]
    p = []
    c_hists = []
    t_hists = []

    for (t,c,sr) in zip(T,C,solver_reward_range)
        #  NORMALIZE THE VECTORS TO GET A BETTER HISTOGRAM
        r_val, r_loc, r_width, r_mom, c_norm, t_norm = residual_moment(c,t,solver_rwd_range=sr)

        ylimits = (minimum([minimum(r_val),0.]), maximum([maximum(r_val),maximum(t_norm.weights),maximum(c_norm.weights)]))

        #  pyplot()
        b1 = bar(r_loc,r_val,bar_width=r_width,label="residual")
        vline!([mean(t)],lw=3,line=:dot,lc=:green,label=L"$r^*$")
        title!("Residual Moment: $(@sprintf("%0.3f",r_mom))\n sr: $sr")
        yaxis!(ylim=ylimits)

        p1 = plot(t_norm,fillalpha=0.4,label="trusted")
        plot!(c_norm,fillalpha=0.4,label="candidate")
        vline!([mean(t)],lw=3,line=:dot,lc=:green,label=L"$r^*$")
        title!("Trusted vs Candidate Reward Probabilities")
        yaxis!(ylim=ylimits)

        plt = plot(p1,b1,size=(1000,900))
        push!(p,plt)
    end
    plot(p[1],p[2],p[3],layout = (3,1))
end
