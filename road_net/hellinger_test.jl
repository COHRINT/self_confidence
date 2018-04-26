using PyPlot, PyCall
using LaTeXStrings
@pyimport mpl_toolkits.axes_grid1 as mpl_kit
ImageGrid = mpl_kit.ImageGrid
@pyimport matplotlib.patches as patches
@pyimport matplotlib.ticker as ticker

include("self_confidence.jl")
pygui(false)

function hellinger_surf(m_rng::Float64,s1::Float64;m2::Float64=0.,s2::Float64=1.)
    return hellinger_normal(m_rng,s1,m2,s2)
end
function sq_surf(m_rng::Float64,f_rng::Float64;s1::Float64=1.,m2::Float64=0.,s2::Float64=1.,return_raw_sq::Bool=false)
    SQ = X3(Normal(m_rng,s1),Normal(m2,s2),global_rwd_range=[f_rng],return_raw_sq=return_raw_sq)
    if return_raw_sq
        return SQ[2]
    else
        return SQ
    end
end

m_rng_sq = linspace(-3.,3.,100)
m_rng = linspace(-5.,5.,100)
s_rng = linspace(0.1,10.,100)
f_rng = linspace(0.1,6.,100)
#
#  h1 = [hellinger_normal(x,1.0,0.,1.0,global_range=20.) for x in m_rng]
#  h2 = [hellinger_normal(x,1.0,0.,1.0,global_range=10.) for x in m_rng]
#  #
#  fig, ax = PyPlot.subplots()
#  #
#  ax[:scatter](m_rng,h1)
#  ax[:scatter](m_rng,h2)

# Solver Quality Surface
fig = plt[:figure](figsize=(6.0,4.0))

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                     nrows_ncols=(1,2),
                     direction="row",
                     aspect = true,
                     axes_pad=0.45,
                     share_all=true,
                     cbar_location="right",
                     cbar_mode="each",
                     cbar_size="3%",
                     cbar_pad=0.10,
                     )

hgs = [sq_surf(x,y,return_raw_sq=true) for x in m_rng_sq, y in f_rng]'

im1 = grid[1][:imshow](hgs,origin="lower",interpolation="bilinear",extent=[-1,1,0.0,2])
grid[1][:set_xlabel](L"$\Delta\mu$")
grid[1][:set_ylabel](L"$f=|\mu_c-\mu_t|/(r_H-r_L)$")
grid[1][:set_title](L"H-dist vs. $\Delta\mu$, and $f$")
grid[1][:cax][:colorbar](im1)

sqs = [sq_surf(x,y,return_raw_sq=false) for x in m_rng_sq, y in f_rng]'

im2 = grid[2][:imshow](sqs,origin="lower",interpolation="bilinear",extent=[-1,1,0.0,2])
grid[2][:set_xlabel](L"$\Delta\mu$")
grid[2][:set_ylabel](L"$f=|\mu_c-\mu_t|/(r_H-r_L)$")
grid[2][:set_title](L"SQ vs. $\Delta\mu$, and $f$")
grid[2][:cax][:colorbar](im2)

savefig("figs/sq_surf.pdf",transparent=true,dpi=300)

# Hellinger Distance Surface
fig2, ax2 = PyPlot.subplots()
fig2[:set_size_inches](6.0,4.0)

hs = [hellinger_surf(x,y) for x in m_rng, y in s_rng]'

cx2 = ax2[:imshow](hs,origin="lower",interpolation="bilinear",extent=[-5,5,0.1,10],clim=[0.,1.])
ax2[:set_aspect]("equal")
fig2[:colorbar](cx2,shrink=1.)
ax2[:set_xlabel](L"$\Delta\mu$")
ax2[:set_ylabel](L"standard deviation, $\sigma$")
ax2[:set_title](L"Hellinger vs. $\Delta\mu$, and $\Delta\sigma$")
#
savefig("figs/hellinger_surf.pdf",transparent=true,dpi=300)

