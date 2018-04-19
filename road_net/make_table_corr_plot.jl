using StatPlots
using CSV

type_list = [Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64]
#  table = CSV.read("logs/test_gconvert.csv")
#  table = CSV.read("logs/net_transition_discount_vary_reference_solver_training.csv",types=type_list,weakrefstrings=false,rows=10)
table = CSV.read("logs/n_vary_reference_solver_training.csv")
#  for c in names(table)
    #  println(typeof(table[c]))
#  end
tbl_ary = convert(Array{Float64},table[:,1:22])

for i = 1:size(tbl_ary,2)
    tbl_ary[:,i] = tbl_ary[:,i]./maximum(tbl_ary[:,i])
end

col_list = []

for i = 1:size(tbl_ary,2)
    if sum(isequal.(diff(tbl_ary[:,i]),0.0))/length(tbl_ary[:,i]) > 0.40
        # this is an (nearly) identical column
        continue
    else
        push!(col_list,i)
    end
end

println(col_list)

cornerplot(tbl_ary[:,col_list],label=names(table)[col_list])
