using Distributions
using GaussianProcesses
using Plots, StatPlots
using LaTeXStrings
include("X3_empirical.jl")

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
max_res = []
for (c,t) in zip(dist_cand,dist_trust)
    # calculate different SQ metrics
    # moment of residuals
    rand_c = rand(c,10000)
    rand_t = rand(t,10000)
    sr = 1.

    _unused, _unused, _unused, rm = residual_moment(rand_c,rand_t,solver_rwd_range=sr,return_hists=false)
    push!(r_moms,rm)

    # moment of expected rwd
    #  display("Mean c: $(mean(rand_c)), Mean t: $(mean(rand_t))")
    me = residual_moment(mean(rand_c),mean(rand_t),solver_rwd_range=sr)
    push!(mom_exps,me)

    # scaled distance of expected rwd
    sd = (mean(rand_c)-mean(rand_t))/sr
    push!(scaled_dists,sd)

    # max residual
    mr = max_residual_moment(rand_c,rand_t,solver_rwd_range=sr,return_hists=false)
    push!(max_res,mr)

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
annotate!(0.07,10.,text(L"SQ\rightarrow High?",:black,:center))

p3 = plot(dist_trust[2],linecolor=:blue,fill=(0,0.2,:blue),legend=false,title="B")
plot!(dist_cand[2],linecolor=:red,fill=(0,0.2,:red))
annotate!(-0.15,12.,text(L"SQ\rightarrow Very Low",:black,:center))

p4 = plot(dist_trust[3],linecolor=:blue,fill=(0,0.2,:blue),legend=false,title="C")
plot!(dist_cand[3],linecolor=:red,fill=(0,0.2,:red))
annotate!(0.15,12.,text(L"SQ\rightarrow Low?",:black,:center))

p5 = plot(dist_trust[4],linecolor=:blue,fill=(0,0.2,:blue),legend=false,title="C")
plot!(dist_cand[4],linecolor=:red,fill=(0,0.2,:red))
annotate!(0.15,12.,text(L"SQ\rightarrow Very High",:black,:center))

# empty filler frames
pe1 = plot(framestyle = :none)
#  annotate!(0.,0.,text("empty",:black,:center))
pe2 = plot(framestyle = :none)
#  annotate!(0.,0.,text("empty",:black,:center))
pe3 = plot(framestyle = :none)
#  annotate!(0.,0.,text("empty",:black,:center))

p1notes = plot(framestyle = :none)
annotate!(0.10,0.5,text("$(@sprintf("residual moment:%0.3f\nmoment of expected rewards:%0.3f\nscaled distance:%0.3f\nmax_res:%0.9f",
                                    r_moms[1],mom_exps[1],scaled_dists[1],max_res[1]))",10,:black,:left,:bottom))
p2notes = plot(framestyle = :none)
annotate!(0.10,0.5,text("$(@sprintf("residual moment:%0.3f\nmoment of expected rewards:%0.3f\nscaled distance:%0.3f\nmax_res:%0.9f",
                                    r_moms[2],mom_exps[2],scaled_dists[2],max_res[2]))",10,:black,:left,:bottom))
p3notes = plot(framestyle = :none)
annotate!(0.10,0.5,text("$(@sprintf("residual moment:%0.3f\nmoment of expected rewards:%0.3f\nscaled distance:%0.3f\nmax_res:%0.9f",
                                    r_moms[3],mom_exps[3],scaled_dists[3],max_res[3]))",10,:black,:left,:bottom))
p4notes = plot(framestyle = :none)
annotate!(0.10,0.5,text("$(@sprintf("residual moment:%0.3f\nmoment of expected rewards:%0.3f\nscaled distance:%0.3f\nmax_res:%0.9f",
                                    r_moms[4],mom_exps[4],scaled_dists[4],max_res[4]))",10,:black,:left,:bottom))

plot(p1,p2,p1notes,pe1,p3,p2notes,pe2,p4,p3notes,pe3,p5,p4notes,layout=(4,3),size=(1000,800))
