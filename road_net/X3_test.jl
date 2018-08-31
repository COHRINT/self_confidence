using Distributions
using GaussianProcesses
using Plots, StatPlots
using LaTeXStrings
#  include("X3_empirical.jl")
include("utilities.jl")
include("self_confidence.jl")

function annotate_pct(coords::Tuple{Float64,Float64},txt::Plots.PlotText,plt::Plots.Plot)
    xmin = plt.subplots[1].attr[:xaxis][:extrema].emin
    xmax = plt.subplots[1].attr[:xaxis][:extrema].emax
    ymin = plt.subplots[1].attr[:yaxis][:extrema].emin
    ymax = plt.subplots[1].attr[:yaxis][:extrema].emax
    #  println("$xmin, $xmax, $ymin, $ymax")
    Plots.annotate!(coords[1]*(xmax-xmin)+xmin,coords[2]*(ymax-ymin)+ymin,txt)
end

srand(12345)
trusted_n=5;                          #number of training points
trusted_x = squeeze([0. 1. rand(1,trusted_n-2)],1);              #predictors
#  trusted_y = ones(trusted_n);    #regressors
trusted_y = zeros(trusted_n);    #regressors

candidate_n=5;
candidate_x = squeeze([0. 1. 0.1464466 rand(1,candidate_n-3)],1);              #predictors
#  candidate_y = (candidate_x-0.5).^2 - 0.125 + 1.;    #regressors
candidate_y = (candidate_x-0.5).^2 - 0.125;    #regressors


#Select mean and covariance function
#  mZero = MeanConst(1.0)
mZero = MeanZero()
kern = SE(-0.5,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)

logObsNoise_trust = -4.0                        # log standard deviation of observation noise (this is optional)
logObsNoise_cand = -4.0                        # log standard deviation of observation noise (this is optional)
trusted_gp = GP(trusted_x,trusted_y,mZero,kern,logObsNoise_trust)       #Fit the GP
candidate_gp = GP(candidate_x,candidate_y,mZero,kern,logObsNoise_cand)       #Fit the GP

#  points_of_interest_x = [0.150,0.5,0.84,0.985]
points_of_interest_x = [0.15,0.5,0.84,0.985]
#  points_of_interest_y = [1.,0.89,1.04,1.116]
points_of_interest_y = [0.,-0.125,0.03,0.116]

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

SQ = []
SQ_raw = []
Ks = []
solver_reward_range = [[-0.,5.], [-0.,0.5], [-0.,0.05], [-0.,0.005]]
for (c,t) in zip(dist_cand,dist_trust)

    sqs = []
    sq_raw = []
    ks = []
    for i in 1:length(solver_reward_range)
        println("#################################")
        (sq,sqr,k) = X3(c,t,global_rwd_range=solver_reward_range[i],return_raw_sq=true)
        println("using $(solver_reward_range[i]) reward range, got $sq")
        push!(sqs,sq)
        push!(sq_raw,sqr)
        push!(ks,k)
    end
    push!(SQ,sqs)
    push!(SQ_raw,sq_raw)
    push!(Ks,ks)
end

show_obs = false
beta_val = 0.999

Plots.scalefontsizes(1.0)

p1 = Plots.plot(trusted_gp,β=beta_val,linecolor=:blue,fillcolor=:blue,lw=3,obsv=show_obs,fillalpha=0.2,label=L"trusted $\mathcal{S}^*$",legend=:topleft)
Plots.plot!(candidate_gp,β=beta_val,linecolor=:red,fillcolor=:red,lw=2,obsv=show_obs,fillalpha=0.2,label=L"candidate $\mathcal{S}$")
Plots.scatter!(points_of_interest_x,points_of_interest_y,marker=:star,markersize=10,label="points of interest")
xlabel!("task parameter")
ylabel!("reward")
for i = 1:length(points_of_interest_x)
    point_label = '@' + i
    Plots.annotate!(points_of_interest_x[i],points_of_interest_y[i],Plots.text("$point_label",:black,:right,:bottom))
end

p2 = Plots.plot(dist_trust[1],linecolor=:blue,fill=(0,0.2,:blue),legend=false,title="A")
Plots.plot!(dist_cand[1],linecolor=:red,fill=(0,0.2,:red))
annotate_pct((0.65,0.8),Plots.text(LaTeXString(@sprintf("\$x_Q(r=5):%0.3f\$\n\$x_Q(r=0.5):%0.3f\$\n\$x_Q(r=0.05):%0.3f\$\n\$x_Q(r=0.005)%0.3f\$",SQ[1][1],SQ[1][2],SQ[1][3],SQ[1][4])),14,:center,:left),p2)
#  annotate_pct((0.65,0.8),Plots.text(LaTeXString(@sprintf("\$x^2\$")),14,:center,:left),p2)
xlabel!("reward")
ylabel!("p(rwd)")
#  annotate_pct((0.8,0.8),text("SQ: $(@sprintf("%0.3f",SQ[1][1]))",:center,:center),p2)
#  annotate!(0.07,10.,text(L"SQ\rightarrow High?",:black,:center))

p3 = Plots.plot(dist_trust[2],linecolor=:blue,fill=(0,0.2,:blue),legend=false,title="B")
Plots.plot!(dist_cand[2],linecolor=:red,fill=(0,0.2,:red))
annotate_pct((0.15,0.8),Plots.text(LaTeXString(@sprintf("\$x_Q(r=5):%0.3f\$\n\$x_Q(r=0.5):%0.3f\$\n\$x_Q(r=0.05):%0.3f\$\n\$x_Q(r=0.005)%0.3f\$",SQ[2][1],SQ[2][2],SQ[2][3],SQ[2][4])),14,:center,:left),p3)
xlabel!("reward")
ylabel!("p(rwd)")
#  annotate_pct((0.8,0.8),text("SQ: $(@sprintf("%0.3f",SQ[2][1]))",:center,:center),p3)
#  annotate!(-0.15,12.,text(L"SQ\rightarrow Very Low",:black,:center))

p4 = Plots.plot(dist_trust[3],linecolor=:blue,fill=(0,0.2,:blue),legend=false,title="C")
Plots.plot!(dist_cand[3],linecolor=:red,fill=(0,0.2,:red))
annotate_pct((0.65,0.8),Plots.text(LaTeXString(@sprintf("\$x_Q(r=5):%0.3f\$\n\$x_Q(r=0.5):%0.3f\$\n\$x_Q(r=0.05):%0.3f\$\n\$x_Q(r=0.005)%0.3f\$",SQ[3][1],SQ[3][2],SQ[3][3],SQ[3][4])),14,:center,:left),p4)
xlabel!("reward")
ylabel!("p(rwd)")
#  annotate_pct((0.8,0.8),text("SQ: $(@sprintf("%0.3f",SQ[3][1]))",:center,:center),p4)
#  annotate!(0.15,12.,text(L"SQ\rightarrow Low?",:black,:center))

p5 = Plots.plot(dist_trust[4],linecolor=:blue,fill=(0,0.2,:blue),legend=false,title="D")
Plots.plot!(dist_cand[4],linecolor=:red,fill=(0,0.2,:red))
annotate_pct((0.65,0.8),Plots.text(LaTeXString(@sprintf("\$x_Q(r=5):%0.3f\$\n\$x_Q(r=0.5):%0.3f\$\n\$x_Q(r=0.05):%0.3f\$\n\$x_Q(r=0.005)%0.3f\$",SQ[4][1],SQ[4][2],SQ[4][3],SQ[4][4])),14,:center,:left),p5)
#  annotate_pct((0.65,0.8),Plots.text(@sprintf("SQ(r=5):%0.3f\nSQ(r=0.5):%0.3f\nSQ(r=0.05):%0.3f\nSQ(r=0.005)%0.3f",SQ[4][1],SQ[4][2],SQ[4][3],SQ[4][4]),14,:center,:left),p5)
xlabel!("reward")
ylabel!("p(rwd)")
#  annotate_pct((0.8,0.8),text("SQ: $(@sprintf("%0.3f",SQ[4][1]))",:center,:center),p5)
#  annotate!(0.15,12.,text(L"SQ\rightarrow Very High",:black,:center))

# empty filler frames
pe1 = Plots.plot(framestyle = :none)
annotate!(0.1,0.5,text("Solver Quality:\n  0->Poor Quality\n  1-> Equal to Trusted\n  2->Better than Trusted\nThink of as fraction of quality\nthresholded at 2",10,:center,:left))
#  annotate!(0.,0.,text("empty",:black,:center))
pe2 = Plots.plot(framestyle = :none)
Plots.annotate!(0.1,0.65,Plots.text("* Calculate mean difference weighted by std",10,:center,:left))
Plots.annotate!(0.5,0.5,Plots.text(L"wd=\frac{m(c)/s(c)-m(t)/s(t)}{range(rwd)}",10,:center,:center))
Plots.annotate!(0.1,0.3,Plots.text("* SQ is a scaled Sigmoid on wd",10,:center,:left))
Plots.annotate!(0.5,0.15,Plots.text(L"SQ=\frac{2exp(wd)}{(exp(wd)+1)}",10,:center,:center))
#  annotate!(0.,0.,text("empty",:black,:center))
pe3 = Plots.plot(framestyle = :none)
Plots.annotate!(0.1,0.65,Plots.text("* In these plots:",10,:center,:left))
Plots.annotate!(0.5,0.5,Plots.text(LaTeXString(@sprintf("range(rwd) = %s",solver_reward_range)),10,:center,:center))
#  annotate!(0.,0.,text("empty",:black,:center))

p1notes = Plots.plot(framestyle = :none)
Plots.annotate!(0.5,0.5,Plots.text(@sprintf("SQ: %0.3f\n%0.3f\n%0.3f\n%0.3f",SQ[1][1],SQ[1][2],SQ[1][3],SQ[1][4]),10,:center,:center))

p2notes = Plots.plot(framestyle = :none)
Plots.annotate!(0.5,0.5,Plots.text("SQ: $(@sprintf("%0.3f",SQ[2][1]))",10,:center,:center))

p3notes = Plots.plot(framestyle = :none)
Plots.annotate!(0.5,0.5,Plots.text("SQ: $(@sprintf("%0.3f",SQ[3][1]))",10,:center,:center))

p4notes = Plots.plot(framestyle = :none)
Plots.annotate!(0.5,0.5,Plots.text("SQ: $(@sprintf("%0.3f",SQ[4][1]))",10,:center,:center))

savefig(p1,"figs/p1.pdf")
savefig(p2,"figs/p2.pdf")
savefig(p3,"figs/p3.pdf")
savefig(p4,"figs/p4.pdf")
savefig(p5,"figs/p5.pdf")
paper_layout = Plots.plot(p2,p3,p4,p5,size=(1000,800))
savefig(paper_layout,"figs/point_compare.pdf")
Plots.plot(p1,p2,pe1,p3,pe2,p4,pe3,p5,layout=(4,2),size=(800,800))
