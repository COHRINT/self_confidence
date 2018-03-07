using Distributions
using GaussianProcesses
using Plots, StatPlots
using LaTeXStrings
include("X3_empirical.jl")
include("utilities.jl")

function annotate_pct(coords::Tuple{Float64,Float64},txt::Plots.PlotText,plt::Plots.Plot)
    xmin = plt.subplots[1].attr[:xaxis][:extrema].emin
    xmax = plt.subplots[1].attr[:xaxis][:extrema].emax
    ymin = plt.subplots[1].attr[:yaxis][:extrema].emin
    ymax = plt.subplots[1].attr[:yaxis][:extrema].emax
    println("$xmin, $xmax, $ymin, $ymax")
    annotate!(coords[1]*(xmax-xmin)+xmin,coords[2]*(ymax-ymin)+ymin,txt)
end

srand(12345)
trusted_n=5;                          #number of training points
trusted_x = squeeze([0. 1. rand(1,trusted_n-2)],1);              #predictors
trusted_y = zeros(trusted_n);    #regressors

candidate_n=5;
candidate_x = squeeze([0. 1. 0.157 rand(1,candidate_n-3)],1);              #predictors
candidate_y = (candidate_x-0.5).^2 - 0.125;    #regressors


#Select mean and covariance function
mZero = MeanZero()                   #Zero mean function
kern = SE(-0.5,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)

logObsNoise_trust = -4.0                        # log standard deviation of observation noise (this is optional)
logObsNoise_cand = -4.0                        # log standard deviation of observation noise (this is optional)
trusted_gp = GP(trusted_x,trusted_y,mZero,kern,logObsNoise_trust)       #Fit the GP
candidate_gp = GP(candidate_x,candidate_y,mZero,kern,logObsNoise_cand)       #Fit the GP

points_of_interest_x = [0.150,0.5,0.78,0.985]
points_of_interest_y = [0.,-0.115,0.,0.116]

μ_trust, σ²_trust = predict_y(trusted_gp,points_of_interest_x);
μ_cand, σ²_cand = predict_y(candidate_gp,points_of_interest_x);

dist_trust = []
for (m,s) in zip(μ_trust,sqrt.(σ²_trust))
    push!(dist_trust,Normal(m,s))
end
dist_cand = []
for (m,s) in zip(μ_cand,sqrt.(σ²_cand))
    push!(dist_cand,Normal(m,s))
end

r_moms = []
mom_exps = []
scaled_dists = []
std_wght = []
solver_reward_range = 0.5
for (c,t) in zip(dist_cand,dist_trust)
    # calculate different SQ metrics
    # moment of residuals
    rand_c = rand(c,10000)
    rand_t = rand(t,10000)

    #  _unused, _unused, _unused, rm = residual_moment(rand_c,rand_t,solver_rwd_range=sr,return_hists=false)
    #  push!(r_moms,rm)
#
    #  # moment of expected rwd
    #  #  display("Mean c: $(mean(rand_c)), Mean t: $(mean(rand_t))")
    #  me = residual_moment(mean(rand_c),mean(rand_t),solver_rwd_range=sr)
    #  push!(mom_exps,me)
#
    #  # scaled distance of expected rwd
    #  sd = (mean(rand_c)-mean(rand_t))/sr
    #  push!(scaled_dists,sd)

    # max residual
    sw = std_weighted_mean_difference(c,t,solver_rwd_range=solver_reward_range)
    push!(std_wght,sw)

end

show_obs = false
beta_val = 0.999

p1 = plot(trusted_gp,β=beta_val,linecolor=:blue,fillcolor=:blue,lw=3,obsv=show_obs,fillalpha=0.2,label="trusted")
plot!(candidate_gp,β=beta_val,linecolor=:red,fillcolor=:red,lw=2,obsv=show_obs,fillalpha=0.2,label="candidate")
scatter!(points_of_interest_x,points_of_interest_y,marker=:star,markersize=10,label="points of interest")
for i = 1:length(points_of_interest_x)
    point_label = '@' + i
    annotate!(points_of_interest_x[i],points_of_interest_y[i],text("$point_label",:black,:right,:bottom))
end

p2 = plot(dist_trust[1],linecolor=:blue,fill=(0,0.2,:blue),legend=false,title="A")
plot!(dist_cand[1],linecolor=:red,fill=(0,0.2,:red))
annotate_pct((0.8,0.8),text("SQ: $(@sprintf("%0.3f",std_wght[1]))",:center,:center),p2)
#  annotate!(0.07,10.,text(L"SQ\rightarrow High?",:black,:center))

p3 = plot(dist_trust[2],linecolor=:blue,fill=(0,0.2,:blue),legend=false,title="B")
plot!(dist_cand[2],linecolor=:red,fill=(0,0.2,:red))
annotate_pct((0.8,0.8),text("SQ: $(@sprintf("%0.3f",std_wght[2]))",:center,:center),p3)
#  annotate!(-0.15,12.,text(L"SQ\rightarrow Very Low",:black,:center))

p4 = plot(dist_trust[3],linecolor=:blue,fill=(0,0.2,:blue),legend=false,title="C")
plot!(dist_cand[3],linecolor=:red,fill=(0,0.2,:red))
annotate_pct((0.8,0.8),text("SQ: $(@sprintf("%0.3f",std_wght[3]))",:center,:center),p4)
#  annotate!(0.15,12.,text(L"SQ\rightarrow Low?",:black,:center))

p5 = plot(dist_trust[4],linecolor=:blue,fill=(0,0.2,:blue),legend=false,title="C")
plot!(dist_cand[4],linecolor=:red,fill=(0,0.2,:red))
annotate_pct((0.8,0.8),text("SQ: $(@sprintf("%0.3f",std_wght[4]))",:center,:center),p5)
#  annotate!(0.15,12.,text(L"SQ\rightarrow Very High",:black,:center))

# empty filler frames
pe1 = plot(framestyle = :none)
annotate!(0.1,0.5,text("Solver Quality:\n  0->Poor Quality\n  1-> Equal to Trusted\n  2->Better than Trusted\nThink of as % quality",:center,:left))
#  annotate!(0.,0.,text("empty",:black,:center))
pe2 = plot(framestyle = :none)
annotate!(0.1,0.5,text("Solver Rwd Range = $solver_reward_range",:center,:left))
#  annotate!(0.,0.,text("empty",:black,:center))
pe3 = plot(framestyle = :none)
#  annotate!(0.,0.,text("empty",:black,:center))

p1notes = plot(framestyle = :none)
annotate!(0.5,0.5,text("SQ: $(@sprintf("%0.3f",std_wght[1]))",:center,:center))

p2notes = plot(framestyle = :none)
annotate!(0.5,0.5,text("SQ: $(@sprintf("%0.3f",std_wght[2]))",:center,:center))

p3notes = plot(framestyle = :none)
annotate!(0.5,0.5,text("SQ: $(@sprintf("%0.3f",std_wght[3]))",:center,:center))

p4notes = plot(framestyle = :none)
annotate!(0.5,0.5,text("SQ: $(@sprintf("%0.3f",std_wght[4]))",:center,:center))

plot(p1,p2,pe1,p3,pe2,p4,pe3,p5,layout=(4,2),size=(1000,600))
