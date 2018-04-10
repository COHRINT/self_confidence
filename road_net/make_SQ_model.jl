using JuliaDB
using MXNet
using PyPlot
using ProgressMeter
using JLD
using LaTeXStrings
include("self_confidence.jl")

mutable struct SQ_model
    input_sch::ML.Schema
    output_sch::ML.Schema
    X3_1net::MXNet.mx.FeedForward #neural network
    X3_2net::MXNet.mx.FeedForward #neural network
    range::Array{Float64}
end

function data_source(batchsize::Int64,train_in,train_out,valid_in,valid_out)
  train = mx.ArrayDataProvider(
    :data => train_in,
    :label => train_out,
    batch_size = batchsize,
    shuffle = true,
    )
  valid = mx.ArrayDataProvider(
    :data => valid_in,
    :label => valid_out,
    batch_size = batchsize,
    shuffle = true,
    )
  println("DONE")
  return train, valid
end

function make_nn_SQ_model(train_fname::String,valid_fname::String,log_fname::String; input_dict::Dict=Dict(),
                          output_dict::Dict=Dict(),nn_epoc::Int64=15,nn_batch_size::Int64=25,
                          train_subsample::Int64=1,valid_subsample::Int64=1)

    training_in, training_out, training_table, input_sch, output_sch = return_data(train_fname,inputs=input_dict,outputs=output_dict,subsample=train_subsample)
    valid_in, valid_out, valid_table, _notused, _notused = return_data(valid_fname,inputs=input_dict,outputs=output_dict,subsample=valid_subsample)

    #  fig,ax = PyPlot.subplots(2,2)
    #  ax[1][:scatter](training_in,training_out[2,:])
    #  ax[2][:scatter](select(training_table,:tprob),select(training_table,:X3_1))
    #  ax[3][:scatter](training_in,training_out[1,:])
    #  ax[4][:scatter](select(training_table,:tprob),select(training_table,:X3_2))
    #  show()

    output_list = collect(keys(output_dict))
    input_list = collect(keys(input_dict))
    info("*****outputs: $output_list\n")
    info("*****inputs: $input_list")

    # create a DNN
    data = mx.Variable(:data)
    label = mx.Variable(:label)
    net = @mx.chain     mx.Variable(:data) =>
                        mx.FullyConnected(num_hidden=10) =>
                        mx.Activation(act_type=:relu) =>
                        mx.FullyConnected(num_hidden=10) =>
                        mx.Activation(act_type=:relu) =>
                        mx.FullyConnected(num_hidden= 1) =>
                        mx.LinearRegressionOutput(mx.Variable(:label))

    # final network definition, don't change, except if using gpu
    X3_1net = mx.FeedForward(net, context=mx.cpu())
    X3_2net = mx.FeedForward(net, context=mx.cpu())

    # set up the optimizer: select one, explore parameters, if desired
    #  optimizer = mx.SGD(η=0.01, μ=0.9, λ=0.00001)
    optimizer = mx.ADAM()

    # get data providers
    # Array(training_out[1,:]') -- this is some crazy conversion so the output is the right type....
    X3_1trainprovider, X3_1evalprovider = data_source(#= batchsize =# nn_batch_size,training_in,Array(training_out[1,:]'),valid_in,Array(valid_out[1,:]'))
    X3_2trainprovider, X3_2evalprovider = data_source(#= batchsize =# nn_batch_size,training_in,Array(training_out[2,:]'),valid_in,Array(valid_out[2,:]'))

    # train, reporting loss for training and evaluation sets
    info("Training X3_1")
    mx.train(X3_1net, optimizer, X3_1trainprovider,
           initializer = mx.NormalInitializer(0.0, 0.1),
           eval_metric = mx.MSE(),
           eval_data = X3_1evalprovider,
           n_epoch = nn_epoc,
           callbacks = [mx.speedometer(),mx.do_checkpoint(string(log_fname,"_net1"),frequency=nn_epoc)])
    info("Training X3_2")
    mx.train(X3_2net, optimizer, X3_2trainprovider,
           initializer = mx.NormalInitializer(0.0, 0.1),
           eval_metric = mx.MSE(),
           eval_data = X3_2evalprovider,
           n_epoch = nn_epoc,
           callbacks = [mx.speedometer(),mx.do_checkpoint(string(log_fname,"_net2"),frequency=nn_epoc)])

    #  println(size(training_out))
    training_range = [minimum(training_out,2) maximum(training_out,2)]
    #  info(minimum(select(training_table,:X3_1)))
    #  info(maximum(select(training_table,:X3_1)))
    #  info(minimum(select(training_table,:X3_2)))
    #  println(maximum(select(training_table,:X3_2)))
    #  error()

    #  println(training_range)
    #  println(output_sch)
    #  error()

    # put into a SQ_model for return
    SQ = SQ_model(input_sch,output_sch,X3_1net,X3_2net,training_range)
    return SQ
end

function SQ_predict(SQ::SQ_model,inputs::Array,outputs::Array;use_eng_units::Bool=false)
    println("inputs:")
    display(inputs)
    println("outputs:")
    display(outputs)
    X3_1plotprovider = mx.ArrayDataProvider(:data => inputs, :label => outputs[2,:], shuffle=false)
    X3_2plotprovider = mx.ArrayDataProvider(:data => inputs, :label => outputs[1,:], shuffle=false)
    X3_1fit = mx.predict(SQ.X3_1net, X3_1plotprovider)
    X3_2fit = mx.predict(SQ.X3_2net, X3_1plotprovider)
    println("Predicted Output:")
    display(X3_1fit)
    display(X3_2fit)
    if use_eng_units
        return restore_eng_units(inputs,SQ.input_sch), restore_eng_units([X3_1fit; X3_2fit],SQ.output_sch)
    else
        return inputs, [X3_1fit; X3_2fit]
    end
end

# convert data back to original units
function restore_eng_units(ary::Array,sch::ML.Schema)
    info("Restoring Engineering Units")
    println(size(ary))
    arrays = Dict()
    i = 1
    for entry in sch
        if entry.second != nothing
            println("key: $(entry.first), value: $(entry.second)")
            entry_mean = mean(entry.second)
            entry_std = std(entry.second)
            arrays[entry.first] = ary[i,:].*entry_std + entry_mean
            i += 1
        end
    end
    return arrays
    #  return [x.second for x in arrays]
end

function return_data(fname::String;inputs::Dict=Dict(),outputs::Dict=Dict(),
                     schema::Array=[],subsample::Int64=1)

    function my_splitschema(xs::ML.Schema,ks::Vector{Symbol})
        return filter((k,v) -> k ∉ ks, xs), filter((k,v) -> k ∈ ks, xs)
    end

    println("Processing $fname")
    table = loadtable(fname,escapechar='"')
    input_sch = nothing
    output_sch = nothing
    schema_dict = Dict()

    if isempty(schema)
        #  p = Progress(length(colnames(table)),dt=0.1,desc="Processing",output=STDOUT)
        for itm_name in colnames(table)
            if itm_name ∈ keys(inputs)
                #  ProgressMeter.next!(p,showvalues=[itm_name])
                key_type = nothing
                if contains(inputs[itm_name],"Continuous")
                    key_type = ML.Continuous
                elseif contains(inputs[itm_name],"Categorical")
                    key_type = ML.Categorical
                end
                schema_dict[itm_name] = key_type
            elseif itm_name ∈ keys(outputs)
                key_type = nothing
                if contains(outputs[itm_name],"Continuous")
                    key_type = ML.Continuous
                elseif contains(outputs[itm_name],"Categorical")
                    key_type = ML.Categorical
                end
                #  ProgressMeter.next!(p,showvalues=[itm_name])
                schema_dict[itm_name] = key_type
            else
                #  ProgressMeter.next!(p,showvalues=[itm_name])
                # ignore variables that aren't inputs or outputs
                schema_dict[itm_name] = nothing
            end
        end

        output_list = collect(keys(outputs))
        input_list = collect(keys(inputs))

        sch = ML.schema(table, hints=schema_dict)
        input_sch, output_sch = my_splitschema(sch,output_list)
    else
        input_sch = schema[1]
        output_sch = schema[2]
    end

    # allow subsampling data, subsample=1 means every data point is used
    # subsample=2 means every other point is used, etc...
    in_data = ML.featuremat(input_sch, table)[:,1:subsample:end]
    out_data = ML.featuremat(output_sch, table)[:,1:subsample:end]

    if any(isnan.(in_data)) || any(isnan.(out_data))
        display(in_data)
        display(out_data)
        error("NaN in data!!!")
    end

    return in_data, out_data, table, input_sch, output_sch
end

function load_network(nn_prefix::String, epoch::Int,sq_fname::String)
    # the default one doesn't work for some reason, so I'll do it by hand
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

function add_sq_annotation(ax::PyCall.PyObject,x_ary::Dict,y_ary::Dict,y_pred_ary::Dict,idx::Int64,xvar::Symbol,yvar_mean::Symbol,yvar_std::Symbol,limits::Dict,txt_loc::Array)
    x_srt = sortperm(tst_in_eng[xvar])
    ex_idx = x_srt[idx]
    ex_x = x_ary[xvar][ex_idx]
    ex_cy = y_ary[yvar_mean][ex_idx]
    ex_ty = y_pred_ary[yvar_mean][ex_idx]
    ex_c = Normal(ex_cy,y_ary[yvar_std][ex_idx])
    ex_t = Normal(ex_ty,y_pred_ary[yvar_std][ex_idx])
    ex_X3 = X3(ex_c,ex_t,global_rwd_range=(limits[yvar_mean][1],limits[yvar_mean][2]))

    ax[:annotate](@sprintf("SQ:%0.2f",ex_X3),xy=[ex_x,ex_cy],xytext=txt_loc,textcoords="figure fraction",arrowprops=Dict(:arrowstyle=>"->"))

end

function scatter_with_conf_bnds(ax::PyCall.PyObject,x::Dict,y::Dict,xval::Symbol,yval::Symbol,yval_std::Symbol,color::Symbol)
    x_srt = sortperm(x[xval])
    ax_ary[:scatter](x[xval][x_srt],y[yval][x_srt],label="true",color=color)
    t_ucl = y[yval][x_srt]+y[yval_std][x_srt]
    t_lcl = y[yval][x_srt]-y[yval_std][x_srt]
    ax_ary[:scatter](x[xval][x_srt],t_ucl,label="model",s=0)
    ax_ary[:scatter](x[xval][x_srt],t_lcl,label="model",s=0)
    ax_ary[:fill_between](x[xval][x_srt],y[yval][x_srt],t_ucl,alpha=0.2,color=color)
    ax_ary[:fill_between](x[xval][x_srt],y[yval][x_srt],t_lcl,alpha=0.2,color=color)

end
function make_label_from_keys(d::Dict)
    z = ""
    for x in collect(keys(d))
        z = string(z,x)
    end
    return z
end

# problem setup
#  train_fname = "logs/transition_vary_reference_solver_training.csv"
#  test_fname = "logs/transition_vary_bad_solver.csv"
#  train_fname = "logs/transition_e_vary_reference_solver_training.csv"
#  test_fname = "logs/transition_e_vary_ok_solver.csv"
train_fname = "logs/net_transition_vary_reference_solver_training.csv"
test_fname = "logs/net_transition_vary_bad_solver.csv"
#  test_fname = "logs/transition_e_vary_mixed_solver.csv"
#  log_fname = "nn_logs/transition_e_vary"
#  log_fname = "nn_logs/transition_vary"
#  inputs = Dict(:tprob=>"ML.Continuous")
#  inputs = Dict(:avg_degree=>"ML.Continuous")
inputs = Dict(:tprob=>"ML.Continuous",:avg_degree=>"ML.Continuous")
outputs = Dict(:X3_1=>"ML.Continuous",:X3_2=>"ML.Continuous")

log_fname = "nn_logs/net_transition_vary_$(make_label_from_keys(inputs))"

first_run = true
num_epoc = 250
training_subsample = 1
if first_run
# make model
    SQmodel = make_nn_SQ_model(train_fname,test_fname,log_fname,input_dict=inputs,output_dict=outputs,nn_epoc=num_epoc,nn_batch_size=150,train_subsample=training_subsample)

    info("Writing variable to file:")
    jldopen(string(log_fname,"_SQmodel.jld"),"w") do file
        JLD.addrequire(file,MXNet)
        JLD.addrequire(file,JuliaDB)
        write(file,"train_fname",train_fname)
        write(file,"test_fname",test_fname)
        write(file,"input_sch",SQmodel.input_sch)
        write(file,"output_sch",SQmodel.output_sch)
        write(file,"inputs",inputs)
        write(file,"outputs",outputs)
        write(file,"range",SQmodel.range)
    end
else
    SQmodel = load_network(log_fname,num_epoc,string(log_fname,"_SQmodel.jld"))
end

# get test and make predictions
test_input, test_output, test_table, input_sch, output_sch = return_data(test_fname, inputs=inputs, outputs=outputs)

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
elseif length(inputs) == 2
    # do 2d stuff
    i1 = collect(keys(inputs))[1]
    i2 = collect(keys(inputs))[2]

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
