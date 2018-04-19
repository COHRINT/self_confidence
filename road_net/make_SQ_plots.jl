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

##### display plots to gui, or just save
## if they plot to gui, they don't have the size specified in teh code
pygui(false)

# problem setup
compare = "ok"
#  net_type = "n_vary"
#  net_type = "transition_vary"
#  net_type = "transition_e_vary"
net_type = "sense_vary"
#  inpts = [:exit_distance]
#  inpts = [:tprob]
inpts = [:sensor_rwd]
#  inpts = [:tprob,:e_mcts]
epocs = 1000
#  sq_example_locations = [1.0,4.0]
#  sq_example_locations = [0.25,0.75]
sq_example_locations = [-15., -150.]

train_fname = "logs/$(net_type)_reference_solver_training.csv"
test_fname = "logs/$(net_type)_$(compare)_solver.csv"

inputs = Dict()
for i in inpts
    inputs[i] = "ML.Continuous"
end
outputs = Dict(:X3_1=>"ML.Continuous",:X3_2=>"ML.Continuous")

log_fname = "$(net_type)_$(make_label_from_keys(inputs))"
log_loc = "nn_logs/"
#  log_path = string(log_loc,"/",log_fname,"_net1-$(epocs)

#  println("##########")
#  println("$log_fname, $(readdir(log_loc))")
#  println("##########")
if !(any([contains(x,string(log_fname,"_")) for x in readdir(log_loc)]))
    println("No nn file exists, making one now")
    #  make_sq_model(net_type,inpts)
    make_sq_model(net_type,inpts,num_epoc=epocs)
end

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

# make figures
if length(inputs) == 1
    fig,ax_ary = PyPlot.subplots(1,1,sharex=false)
    fig[:set_size_inches](8.0,6.0)
    fontsize = 15
    PyPlot.grid()

    i1 = collect(keys(inputs))[1]

    idx1 = nearest_to_x(tst_in_eng[i1],sq_example_locations[1])
    idx2 = nearest_to_x(tst_in_eng[i1],sq_example_locations[2])

    scatter_with_conf_bnds(ax_ary,tst_in_eng,tst_out_eng_ary,i1,:X3_1,:X3_2,:red,subsample=[idx1 idx2],label="candidate",bar=true)
    #  scatter_with_conf_bnds(ax_ary,tst_in_eng,tst_out_eng_ary,i1,:X3_1,:X3_2,:red,subsample=collect(1:length(tst_in_eng[i1])),label="candidate",bar=true)
    #  ax_ary[:scatter](tst_in_eng[i1],tst_out_eng_ary[:X3_1],color=:red)

    ax_ary[:set_xlabel](string(i1),fontsize=fontsize)
    ax_ary[:set_ylabel](string("Reward"),fontsize=fontsize)
    ax_ary[:axhline](limits[:X3_1][2])
    ax_ary[:axhline](limits[:X3_1][1])

    add_sq_annotation(ax_ary,tst_in_eng,tst_out_eng_ary,pred_outputs,idx1,i1,:X3_1,:X3_2,[0.3,0.7],SQmodel,fontsize=fontsize)
    add_sq_annotation(ax_ary,tst_in_eng,tst_out_eng_ary,pred_outputs,idx2,i1,:X3_1,:X3_2,[0.65,0.5],SQmodel,fontsize=fontsize)

    ax_ary[:text](minimum(tst_in_eng[i1]),limits[:X3_1][2],L"r_H",fontsize=fontsize,va="bottom")
    ax_ary[:text](minimum(tst_in_eng[i1]),limits[:X3_1][1],L"r_L",fontsize=fontsize,va="top")

    #  ax_ary[:scatter](tst_in_eng[i1],pred_outputs[:X3_1],color=:blue)
    scatter_with_conf_bnds(ax_ary,tst_in_eng,pred_outputs,i1,:X3_1,:X3_2,:blue,subsample=collect(1:8:length(tst_in_eng[i1])),label="trusted",bar=false)
    PyPlot.legend()
elseif length(inputs) > 1
    PyPlot.using3D()
    fig,ax_ary = PyPlot.subplots(1,1)
    fig[:set_size_inches](8.0,4.0)
    ax_ary = PyPlot.subplot(111,projection="3d")
    fontsize = 15

    # make correlation plots
    i1 = inpts[1]
    i2 = inpts[2]

    #  corrplot(data_mat)
    poi1 = [0.2;200.] # point of interest, where X3 will be calculated
    poi2 = [0.7;800.] # point of interest, where X3 will be calculated
    poi1_norm = [(poi1[1]-mean(input_sch[i1]))/std(input_sch[i1]);(poi1[2]-mean(input_sch[i2]))/std(input_sch[i2])]
    poi2_norm = [(poi2[1]-mean(input_sch[i1]))/std(input_sch[i1]);(poi2[2]-mean(input_sch[i2]))/std(input_sch[i2])]
    subsample_num = 3

    ax_ary[:scatter3D](tst_in_eng[i1][1:subsample_num:end],tst_in_eng[i2][1:subsample_num:end],tst_out_eng_ary[:X3_1][1:subsample_num:end],color=:red,alpha=0.2,label="Candidate")
    ax_ary[:scatter3D](tst_in_eng[i1][1:subsample_num:end],tst_in_eng[i2][1:subsample_num:end],pred_outputs[:X3_1][1:subsample_num:end],color=:blue,alpha=0.2,label="Trusted")

    add_sq_scatter3d_annotation(ax_ary,test_input,tst_out_eng_ary,i1,i2,poi1_norm,SQmodel,marker="o",s=100,fontsize=fontsize)
    add_sq_scatter3d_annotation(ax_ary,test_input,tst_out_eng_ary,i1,i2,poi2_norm,SQmodel,marker="*",s=100,fontsize=fontsize)

    ax_ary[:view_init](azim=-76,elev=25)

    ax_ary[:set_xlabel](string(i1),fontsize=fontsize)
    ax_ary[:set_ylabel](string(i2),fontsize=fontsize)
    ax_ary[:set_zlabel](string("Reward"),fontsize=fontsize)

    PyPlot.legend()
else
 error("can't support more than 2 inputs yet")
end

#  show()
PyPlot.savefig(string(log_fname,"_",compare,".png"),dpi=300,transparent=true)
