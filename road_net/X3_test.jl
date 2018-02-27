using Distributions
using GaussianProcesses
using Plots, StatPlots

srand(1234)
trusted_n=5;                          #number of training points
trusted_x = squeeze([0. 1. 0.16 rand(1,trusted_n-3)],1);              #predictors
trusted_y = zeros(trusted_n);    #regressors

candidate_n=5;
candidate_x = squeeze([0. 1. rand(1,candidate_n-2)],1);              #predictors
candidate_y = (candidate_x-0.5).^2 - 0.125;    #regressors


#Select mean and covariance function
mZero = MeanZero()                   #Zero mean function
kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)

logObsNoise_trust = -3.5                        # log standard deviation of observation noise (this is optional)
logObsNoise_cand = -4.0                        # log standard deviation of observation noise (this is optional)
trusted_gp = GP(trusted_x,trusted_y,mZero,kern,logObsNoise_trust)       #Fit the GP
candidate_gp = GP(candidate_x,candidate_y,mZero,kern,logObsNoise_cand)       #Fit the GP

points_of_interest_x = [0.16,0.5,0.975]
points_of_interest_y = [0.,-0.126,0.1]

μ_trust, σ²_trust = predict_y(trusted_gp,points_of_interest_x);
μ_cand, σ²_cand = predict_y(candidate_gp,points_of_interest_x);

dist_trust = []
for (m,v) in zip(μ_trust,sqrt.(σ²_trust))
    push!(dist_trust,Normal(m,v))
end
dist_cand = []
for (m,v) in zip(μ_cand,sqrt.(σ²_cand))
    push!(dist_cand,Normal(m,v))
end

p1 = plot(trusted_gp,β=0.999,linecolor=:blue,fillcolor=:blue,lw=3,obsv=false,fillalpha=0.2,label="trusted")
plot!(candidate_gp,β=0.999,linecolor=:red,fillcolor=:red,lw=2,obsv=false,fillalpha=0.2,label="candidate")
scatter!(points_of_interest_x,points_of_interest_y,marker=:star,markersize=10,label="points of interest")

p2 = plot(dist_trust[1],linecolor=:blue,fill=(0,0.2,:blue))
plot!(dist_cand[1],linecolor=:red,fill=(0,0.2,:red))

p3 = plot(dist_trust[2],linecolor=:blue,fill=(0,0.2,:blue))
plot!(dist_cand[2],linecolor=:red,fill=(0,0.2,:red))

p4 = plot(dist_trust[3],linecolor=:blue,fill=(0,0.2,:blue))
plot!(dist_cand[3],linecolor=:red,fill=(0,0.2,:red))

plot(p1,p2,p3,p4,size=(1200,800))
