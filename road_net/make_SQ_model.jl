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
inputs = Dict(:tprob=>"ML.Continuous",:E=>"ML.Continuous")
outputs = Dict(:X3_1=>"ML.Continuous",:X3_2=>"ML.Continuous")

log_fname = "nn_logs/net_transition_vary_$(make_label_from_keys(inputs))"

num_epoc = 250
training_subsample = 1
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
