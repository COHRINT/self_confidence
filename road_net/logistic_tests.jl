using PyPlot
using LaTeXStrings

function general_logistic(x::Array{Float64};k::Float64=1.,x0::Float64=0.,L::Float64=1.)
    return L./(1+exp.(-k.*(x-x0)))
end

x = collect(linspace(-20.,20.,500))

ks = [1/3,1/2,1.,2.]
ys = zeros(length(x),length(ks))
i = 1
for k in ks
    ys[:,i] = general_logistic(x,L=1.,k=k)
    i += 1
end

fig = PyPlot.figure()
fig[:set_size_inches](8.,6.)
ax = fig[:add_subplot](1,1,1)
fontsize = 12
ax[:set_title](LaTeXString("Location of \$x_{sat}\$ for different \$k\$\n\$x_{sat}\\approx 5/k\$, or \$k\\approx 5/x_{sat}\$"),size=fontsize)
ax[:set_xlabel]("x",size=fontsize)
ax[:set_ylabel]("y",size=fontsize)

for i = 1:length(ks)
    k = ks[i]
    k_str = LaTeXString(@sprintf("\$k=%0.2f\$",k))
    x_str = LaTeXString(@sprintf("\$x_{sat}\\approx %0.2f\$",5/k))
    lbl = LaTeXString("$k_str, $x_str")
    #  println(lbl)
    ax[:plot](x,ys[:,i],label=lbl)
    ax[:axvline](x=[5/k],lw=1,color=:black,label="")
end
fig[:tight_layout]()
ax[:legend]()
show()
PyPlot.savefig("figs/logistic_saturation.pdf",dpi=300,transparent=true)

# conclusions, x_saturation (the x at which the logistic reaches its limit, is approximately equal to 5*k, or choose k=5/x_sat for saturation at x_sat
