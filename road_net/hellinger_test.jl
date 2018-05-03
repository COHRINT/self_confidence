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
function sq_surf(m_rng::Float64,f_rng::Float64;s1::Float64=1.,m2::Float64=0.,s2::Float64=1.,return_raw_sq::Bool=false,alpha=1/2)
    SQ = X3(Normal(m_rng,s1),Normal(m2,s2),global_rwd_range=[f_rng],return_raw_sq=return_raw_sq,alpha=alpha)
    if return_raw_sq
        return SQ[2]
    else
        return SQ
    end
end

sq_rng_m = [-3.;3.]
sq_rng_f = [0.1;2.]
ar = diff(sq_rng_m)./diff(sq_rng_f)[1]

m_rng_sq = linspace(sq_rng_m[1],sq_rng_m[2],100)
m_rng = linspace(-5.,5.,100)
s_rng = linspace(0.1,10.,100)
f_rng = linspace(sq_rng_f[1],sq_rng_f[2],100)

# Solver Quality Surface
if false
    fig = plt[:figure](figsize=(6.0,8.0))

    grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                         nrows_ncols=(3,2),
                         direction="row",
                         aspect = true,
                         axes_pad=0.65,
                         share_all=true,
                         cbar_location="right",
                         cbar_mode="each",
                         cbar_size="3%",
                         cbar_pad=0.10,
                         )

    as = [1.,1/2,1/8]
    hgs_a1 = [sq_surf(x,y,return_raw_sq=true,alpha=as[1]) for x in m_rng_sq, y in f_rng]'
    hgs_a2 = [sq_surf(x,y,return_raw_sq=true,alpha=as[2]) for x in m_rng_sq, y in f_rng]'
    hgs_a3 = [sq_surf(x,y,return_raw_sq=true,alpha=as[3]) for x in m_rng_sq, y in f_rng]'

    sq_cmap = :viridis
    cont_cmap = :binary
    im1 = grid[1][:imshow](hgs_a1,origin="lower",cmap=sq_cmap,interpolation="bilinear",extent=[sq_rng_m;sq_rng_f])
    grid[1][:contour](m_rng_sq,f_rng,hgs_a1,cmap=cont_cmap,linewidths=0.5)
    grid[1][:set_ylabel](L"f, \alpha=1")
    grid[1][:set_title](L"sq vs. $\Delta\mu$, and $f$")
    grid[1][:cax][:colorbar](im1)
    grid[1][:set_aspect](ar)

    im3 = grid[3][:imshow](hgs_a2,origin="lower",cmap=sq_cmap,interpolation="bilinear",extent=[sq_rng_m;sq_rng_f])
    grid[3][:contour](m_rng_sq,f_rng,hgs_a2,cmap=cont_cmap,linewidths=0.5)
    grid[3][:set_ylabel](L"f,\alpha=1/2")
    grid[3][:cax][:colorbar](im3)
    grid[3][:set_aspect](ar)

    im5 = grid[5][:imshow](hgs_a3,origin="lower",cmap=sq_cmap,interpolation="bilinear",extent=[sq_rng_m;sq_rng_f])
    grid[5][:contour](m_rng_sq,f_rng,hgs_a3,cmap=cont_cmap,linewidths=0.5)
    grid[5][:set_xlabel](L"\Delta\mu")
    grid[5][:set_ylabel](L"f,\alpha=1/8")
    grid[5][:cax][:colorbar](im5)
    grid[5][:set_aspect](ar)

    sqs_a1 = [sq_surf(x,y,return_raw_sq=false,alpha=as[1]) for x in m_rng_sq, y in f_rng]'
    sqs_a2 = [sq_surf(x,y,return_raw_sq=false,alpha=as[2]) for x in m_rng_sq, y in f_rng]'
    sqs_a3 = [sq_surf(x,y,return_raw_sq=false,alpha=as[3]) for x in m_rng_sq, y in f_rng]'

    im2 = grid[2][:imshow](sqs_a1,origin="lower",cmap=sq_cmap,interpolation="bilinear",extent=[sq_rng_m;sq_rng_f],vmin=0.,vmax=2)
    grid[2][:contour](m_rng_sq,f_rng,sqs_a1,cmap=cont_cmap,linewidths=0.5)
    grid[2][:set_title](L"SQ vs. $\Delta\mu$, and $f$")
    grid[2][:cax][:colorbar](im2)
    grid[2][:set_aspect](ar)
    im4 = grid[4][:imshow](sqs_a2,origin="lower",cmap=sq_cmap,interpolation="bilinear",extent=[sq_rng_m;sq_rng_f],vmin=0.,vmax=2)
    grid[4][:contour](m_rng_sq,f_rng,sqs_a2,cmap=cont_cmap,linewidths=0.5)
    grid[4][:cax][:colorbar](im4)
    grid[4][:set_aspect](ar)
    im6 = grid[6][:imshow](sqs_a3,origin="lower",cmap=sq_cmap,interpolation="bilinear",extent=[sq_rng_m;sq_rng_f],vmin=0.,vmax=2)
    grid[6][:contour](m_rng_sq,f_rng,sqs_a3,cmap=cont_cmap,linewidths=0.5)
    grid[6][:set_xlabel](L"$\Delta\mu$")
    grid[6][:cax][:colorbar](im6)
    grid[6][:set_aspect](ar)

    savefig("figs/sq_surf.pdf",transparent=true,dpi=300)
end

if true
    # Hellinger Distance Surface
    fig2, ax2 = PyPlot.subplots()
    fig2[:set_size_inches](4.0,4.0)

    hs = [hellinger_surf(x,y) for x in m_rng, y in s_rng]'

    cx2 = ax2[:imshow](hs,origin="lower",interpolation="bilinear",extent=[-5,5,0.1,10],clim=[0.,1.])
    ax2[:set_aspect]("equal")
    fig2[:colorbar](cx2,shrink=0.73)
    ax2[:set_xlabel](L"$\Delta\mu$")
    ax2[:set_ylabel](L"$\sigma_1$")
    ax2[:set_title](L"Hellinger vs. $\Delta\mu$, and $\sigma_1$")
    PyPlot.tight_layout()
    #
    savefig("figs/hellinger_surf.pdf",transparent=true,dpi=300)
end
