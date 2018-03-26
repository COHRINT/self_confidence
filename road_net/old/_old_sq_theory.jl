using Plots, StatPlots
using Distributions
include("self_confidence.jl")

srand(12345)

m = -0.125:0.01:0.125
s = 0.01:0.1:0.125

f(x,y)= begin
    x/y
end

p1 = contour(m,s,f,levels=50,fill=true)
ylabel!("std dev")
xlabel!("mean")

c = []
t = []
for i = 1:9
    push!(c,Normal(randn(1)[1],rand(1)[1]*2.))
    push!(t,Normal(randn(1)[1],rand(1)[1]*2.))
end

q(x,y) = begin
    val = (x.μ/x.σ - y.μ/y.σ)/1. # the denominator is to scale by the total problem reward range
    return exp(val)/(exp(val)+1.) - 0.
end

p = []
for (z,r) in zip(c,t)
    plt = plot(r,label="trusted")
    plot!(z,label="candidate")
    #  plt.subplots[1].o[:annotate](xytext=text(@sprintf("SQ: %0.3E",q(z,r))),textcoords="axes fraction")
    #  ax[:annotate]("",
                  #  xytext=(0.5,0.5),
                  #  xy=(0.5,0.5),
                  #  textcoords="axes fraction",
                  #  zorder=999)
    xmin = plt.subplots[1].attr[:xaxis][:extrema].emin
    xmax = plt.subplots[1].attr[:xaxis][:extrema].emax
    ymin = plt.subplots[1].attr[:yaxis][:extrema].emin
    ymax = plt.subplots[1].attr[:yaxis][:extrema].emax
    SQ_val = @sprintf("SQ: %0.3f",q(z,r))
    annotate!(0.75*(xmax-xmin)+xmin,0.75*(ymax-ymin)+ymin,text(SQ_val,10,:black,:center))
    push!(p,plt)
end

plot(p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],size=(900,900))
