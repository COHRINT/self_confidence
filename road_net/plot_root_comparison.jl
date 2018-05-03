using PyPlot
using LaTeXStrings

x = collect(linspace(0.,1.,1000))
y(r) = x.^r
rs = [1/4,1/3,1/2,2/3,3/4,1.]

fig = plt[:figure](figsize=(5.0,4.0))
ax = fig[:subplots](1,1)
fs = 10

for r in rs
    lw = 1
    if r == 1/2
        lw = 3
    end
    ax[:plot](x,y(r),label="$(@sprintf("r=%.2f",r))",lw = lw)
end

ax[:set_title](L"$y=x^r$ for different $r$",fontsize=fs)
ax[:set_xlabel](L"x",fontsize=fs)
ax[:set_ylabel](L"y=x^r",fontsize=fs)
ax[:legend]()

savefig("figs/power_comparison.pdf",dpi=300,transparent=true)
