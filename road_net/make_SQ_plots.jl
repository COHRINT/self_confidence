using PyPlot
include("make_SQ_model.jl")

function load_network(nn_prefix::String, epoch::Int,sq_fname::String)
    # the default one doesn't work for some reason, so I'll do it by hand
    println(nn_prefix)
    arch1, arg_params1, aux_params1 = mx.load_checkpoint(string(nn_prefix,"_net1"),epoch)
    arch2, arg_params2, aux_params2 = mx.load_checkpoint(string(nn_prefix,"_net2"),epoch)

    net1 = mx.FeedForward(arch1)
    net1.arg_params = arg_params1
    net1.aux_params = aux_params1

    net2 = mx.FeedForward(arch2)
    net2.arg_params = arg_params2
    net2.aux_params = aux_params2

    sq_data = JLD.load(sq_fname)
    input_sch = sq_data["input_sch"]
    output_sch = sq_data["output_sch"]
    range = sq_data["range"]

    SQ = SQ_model(input_sch,output_sch,net1,net2,range)
    return SQ
end
function make_label_from_keys(d::Dict)
    z = ""
    for x in collect(keys(d))
        z = string(z,x)
    end
    return z
end
function searchdir(path,key1)
    # list of files matching key1
    filt_list = filter(x->contains(x,key1),readdir(path))

    return filt_list
end
function searchdir(path,key1,key2)
    # list of files matching key1
    filt_list = searchdir(path,key1)

    # subset of that list that contains key2
    filt_list2 = filter(x->contains(x,key2),filt_list)

    return filt_list2
end

# problem setup
net_type = "net_transition_discount_vary"

train_fname = "logs/$(net_type)_reference_solver_training.csv"
test_fname = "logs/$(net_type)_bad_solver.csv"

inputs = Dict(:tprob=>"ML.Continuous",:discount=>"ML.Continuous")
outputs = Dict(:X3_1=>"ML.Continuous",:X3_2=>"ML.Continuous")

log_fname = "$(net_type)_$(make_label_from_keys(inputs))"
log_loc = "nn_logs/"

param_files = searchdir(log_loc,log_fname,".params")

num_epocs = parse(split(match(r"-\d+",param_files[1]).match,"-")[2])

SQmodel = load_network(string(log_loc,log_fname),num_epocs,string(log_loc,log_fname,"_SQmodel.jld"))

# get test and make predictions
test_input, test_output, test_table, input_sch, output_sch = return_data(test_fname, inputs=inputs, outputs=outputs)

data_mat = ML.featuremat(merge(input_sch,output_sch),test_table)

info("restoring limits")
limits = restore_eng_units(SQmodel.range,SQmodel.output_sch)
info("getting predictions")
_notused, pred_outputs = SQ_predict(SQmodel,test_input,test_output,use_eng_units=true)

info("restoring test data")
tst_in_eng = restore_eng_units(test_input,input_sch)
tst_out_eng_ary = restore_eng_units(test_output,output_sch)
#  tst_out_eng = tst_out_eng_ary[:X3_1]
#  tst_out_eng_2 = tst_out_eng_ary[:X3_2]

# make figures

if length(inputs) == 1
 fig,ax_ary = PyPlot.subplots(1,1,sharex=false)
 fig[:set_size_inches](8.0,6.0)

 i1 = collect(keys(inputs))[1]

 scatter_with_conf_bnds(ax_ary,tst_in_eng,tst_out_eng_ary,i1,:X3_1,:X3_2,:red)

 ax_ary[:set_xlabel](string(i1))
 ax_ary[:set_ylabel](string(:X3_1))
 ax_ary[:axhline](limits[:X3_1][2])
 ax_ary[:axhline](limits[:X3_1][1])

 add_sq_annotation(ax_ary,tst_in_eng,tst_out_eng_ary,pred_outputs,50,i1,:X3_1,:X3_2,limits,[0.3,0.7])
 add_sq_annotation(ax_ary,tst_in_eng,tst_out_eng_ary,pred_outputs,200,i1,:X3_1,:X3_2,limits,[0.65,0.5])
 ax_ary[:text](0.5,limits[:X3_1][2],L"r_H",fontsize=15,va="bottom")
 ax_ary[:text](0.5,limits[:X3_1][1],L"r_L",fontsize=15,va="top")

 scatter_with_conf_bnds(ax_ary,tst_in_eng,pred_outputs,i1,:X3_1,:X3_2,:blue)
elseif length(inputs) > 1
    # make correlation plots
    i1 = collect(keys(inputs))[1]
    i2 = collect(keys(inputs))[2]

    #  corrplot(data_mat)

    scatter3D(tst_in_eng[i1],tst_in_eng[i2],tst_out_eng_ary[:X3_1],color=:red,alpha=0.2)
    scatter3D(tst_in_eng[i1],tst_in_eng[i2],pred_outputs[:X3_1],color=:blue,alpha=0.2)

    #  t_ucl = y[yval][x_srt]+y[yval_std][x_srt]
    #  t_lcl = y[yval][x_srt]-y[yval_std][x_srt]
    #  ax_ary[:scatter](x[xval][x_srt],t_ucl,label="model",s=0)
    #  ax_ary[:scatter](x[xval][x_srt],t_lcl,label="model",s=0)
    #  ax_ary[:fill_between](x[xval][x_srt],y[yval][x_srt],t_ucl,alpha=0.2,color=color)
    #  ax_ary[:fill_between](x[xval][x_srt],y[yval][x_srt],t_lcl,alpha=0.2,color=color)

    #  ax_ary[:set_xlabel](string(i1))
    #  ax_ary[:set_ylabel](string(i2))
    #  ax_ary[:set_zlabel](string(:X3_1))
    #  ax_ary[:axhline](limits[:X3_1][2])
    #  ax_ary[:axhline](limits[:X3_1][1])
else
 error("can't support more than 2 inputs yet")
end

#  PyPlot.legend()
show()
# 3d scatter

#  plot(p1,p3,p4,p5,size=(1200,600))
#  plot(p1,size=(900,600))
