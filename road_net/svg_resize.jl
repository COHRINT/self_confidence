include("utilities.jl")

fdir = "./imgs"
files = searchdir(fdir,".svg")
mod_imgs_fldr = "./imgs/squared"
function parse_strings(s_ary)
    ary = Array{Float64}(length(s_ary))
    for i = 1:length(s_ary)
        ary[i] = parse(Float64,s_ary[i])
    end
    return ary
end

for f in files
    txt = open(joinpath(fdir,f)) do file
        readlines(file)
    end
    loi = txt[2] # loi -> line of interest

    # need to find maximum of width and height, then cange the smaller to match it
    # then offset the viewBox to keep the image in the center

    #capture number with 'pt' at the end, this will be width and height
    r1 = r"(-?\d+(?>\.\d+)?)pt.+?(-?\d+(?>\.\d+)?)pt"
    # capture 4 number sequence, this is the 'viewBox' stuff
    r2 = r"(-?\d+(?>\.\d+)?) (-?\d+(?>\.\d+)?) (-?\d+(?>\.\d+)?) (-?\d+(?>\.\d+)?)"

    w_h = match(r1,loi)
    wh = parse_strings(w_h.captures)
    box = match(r2,loi)
    vb = parse_strings(box.captures)

    ind_max_d = indmax(wh)
    ind_min_d = indmin(wh)
    dmax = wh[ind_max_d]
    dmin = wh[ind_min_d]
    if dmax == dmin
        # dims already square
        continue
    end
    wh_diff = dmax - dmin
    wh[ind_min_d] = wh[ind_max_d]

    if ind_min_d == 1
        #we need to  modify x
        vb[1] -= wh_diff/2
        vb[3] += wh_diff/2
    else
        # we need to modify y
        vb[2] -= wh_diff/2
        vb[4] += wh_diff/2
    end

    mod_loi = loi
    mod_loi = replace(mod_loi,w_h.match,@sprintf("%0.3fpt\" height=\"%0.3fpt",wh[1],wh[2]))

    mod_loi = replace(mod_loi,box.match,@sprintf("%0.3f %0.3f %0.3f %0.3f",vb[1],vb[2],vb[3],vb[4]))

    println(mod_loi)

    txt[2] = mod_loi

    open("$mod_imgs_fldr/$f","w") do f
        writedlm(f,txt,'\n',quotes=false)
    end

end
