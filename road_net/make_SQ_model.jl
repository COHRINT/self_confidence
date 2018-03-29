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
    network::MXNet.mx.FeedForward #neural network
    range::Array{Float64}
end

function data_source(batchsize::Int64,train_in,train_out,valid_in,valid_out)
    #  println(batchsize)
    #  display(train_in)
    #  display(train_out)
    #  display(valid_in)
    #  display(valid_out)
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
  train, valid
end

function make_nn_SQ_model(train_fname::String,valid_fname::String,log_fname::String; input_dict::Dict=Dict(),
                          output_dict::Dict=Dict(),nn_epoc::Int64=15,nn_batch_size::Int64=25)

    training_in, training_out, training_table, input_sch, output_sch = return_data(train_fname,inputs=input_dict,outputs=output_dict)
    valid_in, valid_out, valid_table, _notused, _notused = return_data(valid_fname,inputs=input_dict,outputs=output_dict)

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
                        mx.FullyConnected(num_hidden= length(output_list)) =>
                        mx.LinearRegressionOutput(mx.Variable(:label))

    # final network definition, don't change, except if using gpu
    network = mx.FeedForward(net, context=mx.cpu())

    # set up the optimizer: select one, explore parameters, if desired
    #  optimizer = mx.SGD(η=0.01, μ=0.9, λ=0.00001)
    optimizer = mx.ADAM()

    # get data providers
    trainprovider, evalprovider = data_source(#= batchsize =# nn_batch_size,training_in,training_out,
                                              valid_in,valid_out)


    # train, reporting loss for training and evaluation sets
    mx.train(network, optimizer, trainprovider,
           initializer = mx.NormalInitializer(0.0, 0.1),
           eval_metric = mx.MSE(),
           eval_data = evalprovider,
           n_epoch = nn_epoc,
           callbacks = [mx.speedometer(),mx.do_checkpoint(log_fname,frequency=2)])

    println(size(training_out))
    training_range = [minimum(training_out,2) maximum(training_out,2)]

    # put into a SQ_model for return
    SQ = SQ_model(input_sch,output_sch,network,training_range)
    return SQ
end

function SQ_predict(SQ::SQ_model,inputs::Array,outputs::Array;use_eng_units::Bool=false)
    println("inputs:")
    display(inputs)
    println("outputs:")
    display(outputs)
    plotprovider = mx.ArrayDataProvider(:data => inputs, :label => outputs, shuffle=false)
    fit = mx.predict(SQ.network, plotprovider)
    println("Predicted Output:")
    display(fit)
    if use_eng_units
        return restore_eng_units(inputs,SQ.input_sch), restore_eng_units(fit,SQ.output_sch)
    else
        return inputs, fit
    end
end

# convert data back to original units
function restore_eng_units(ary::Array,sch::ML.Schema)
    info("Restoring Engineering Units")
    arrays = Dict()
    i = 1
    for entry in sch
        if entry.second != nothing
            println("key: $(entry.first), value: $(entry.second)")
            entry_mean = mean(entry.second)
            entry_std = std(entry.second)
            println("mean: $entry_mean, std: $entry_std")
            arrays[entry.first] = ary[i,:].*entry_std + entry_mean
            i += 1
        end
    end
    return arrays
    #  return [x.second for x in arrays]
end

function return_data(fname::String;inputs::Dict=Dict(),outputs::Dict=Dict(),
                     schema::Array=[])

    function my_splitschema(xs::ML.Schema,ks::Vector{Symbol})
        return filter((k,v) -> k ∉ ks, xs), filter((k,v) -> k ∈ ks, xs)
    end

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

    in_data = ML.featuremat(input_sch, table)
    out_data = ML.featuremat(output_sch, table)

    if any(isnan.(in_data)) || any(isnan.(out_data))
        error("NaN in data!!!")
    end

    return in_data, out_data, table, input_sch, output_sch
end

function load_network(nn_prefix::String, epoch::Int,sq_fname::String)
    # the default one doesn't work for some reason, so I'll do it by hand
    arch, arg_params, aux_params = mx.load_checkpoint(nn_prefix,epoch)
    model = mx.FeedForward(arch)
    model.arg_params = arg_params
    model.aux_params = aux_params

    sq_data = JLD.load(sq_fname)
    input_sch = sq_data["input_sch"]
    output_sch = sq_data["output_sch"]
    range = sq_data["range"]

    SQ = SQ_model(input_sch,output_sch,model,range)
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

    ax[:annotate](@sprintf("SQ:%0.2f",ex1_X3),xy=[ex_x,ex_cy],xytext=txt_loc,textcoords="figure fraction",arrowprops=Dict(:arrowstyle=>"->"))

end

# problem setup
#  train_fname = "logs/transition_vary_4.csv"
#  test_fname = "logs/transition_vary_test_4.csv"
train_fname = "logs/transition_e_vary_reference_solver_training.csv"
test_fname = "logs/transition_e_vary_bad_solver.csv"
#  test_fname = "logs/transition_e_vary_mixed_solver.csv"
log_fname = "nn_logs/transition_e_vary"
#  log_fname = "nn_logs/transition_vary_4"
inputs = Dict(:tprob=>"ML.Continuous")
#  inputs = Dict(:tprob=>"ML.Continuous",:e_mcts=>"ML.Continuous")
#  outputs = Dict(:X3_1=>"ML.Continuous")
outputs = Dict(:X3_1=>"ML.Continuous",:X3_2=>"ML.Continuous")

first_run = false
num_epoc = 1500
if first_run
# make model
    SQmodel = make_nn_SQ_model(train_fname,test_fname,log_fname,input_dict=inputs,output_dict=outputs,nn_epoc=num_epoc,nn_batch_size=250)

    info("Writing variable to file:")
    jldopen("test_SQmodel.jld","w") do file
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
    SQmodel = load_network(log_fname,num_epoc,"test_SQmodel.jld")
end

# get test and make predictions
test_input, test_output, test_table, input_sch, output_sch = return_data(test_fname, inputs=inputs, outputs=outputs)
_notused, pred_outputs = SQ_predict(SQmodel,test_input,test_output,use_eng_units=true)

tst_in_eng = restore_eng_units(test_input,input_sch)
tst_out_eng_ary = restore_eng_units(test_output,output_sch)
limits = restore_eng_units(SQmodel.range,output_sch)
tst_out_eng = tst_out_eng_ary[:X3_1]
tst_out_eng_2 = tst_out_eng_ary[:X3_2]


function scatter_with_conf_bnds(ax::PyCall.PyObject,x::Dict,y::Dict,xval::Symbol,yval::Symbol,yval_std::Symbol,color::Symbol)
    x_srt = sortperm(x[xval])
    ax_ary[:scatter](x[:tprob][x_srt],y[:X3_1][x_srt],label="true",color=color)
    t_ucl = y[yval][x_srt]+y[yval_std][x_srt]
    t_lcl = y[yval][x_srt]-y[yval_std][x_srt]
    ax_ary[:scatter](x[xval][x_srt],t_ucl,label="model",s=0)
    ax_ary[:scatter](x[xval][x_srt],t_lcl,label="model",s=0)
    ax_ary[:fill_between](x[xval][x_srt],y[yval][x_srt],t_ucl,alpha=0.2,color=color)
    ax_ary[:fill_between](x[xval][x_srt],y[yval][x_srt],t_lcl,alpha=0.2,color=color)

end
# make figures
if false
    fig,ax_ary = PyPlot.subplots(2,1,sharex=false)
    fig[:set_size_inches](8.0,8.0)

    for k in [:X3_1]
        # actual invputs/outputs vs predicted inputs/outputs
        ax_ary[1][:scatter](tst_in_eng[:tprob],tst_out_eng_ary[k],label="true")
        ax_ary[1][:scatter](tst_in_eng[:tprob],tst_out_eng_ary[k]+tst_out_eng_ary[:X3_2],label="model")
        ax_ary[1][:set_xlabel]("tprob")
        ax_ary[1][:set_ylabel](string(k))
        ax_ary[1][:scatter](tst_in_eng[:tprob],pred_outputs[k],label="model")
        ax_ary[1][:scatter](tst_in_eng[:tprob],pred_outputs[k]+pred_outputs[:X3_2],label="model")

        # actual output vs predicted, 45 deg line is desirable
        ax_ary[2][:scatter](pred_outputs[k],tst_out_eng_ary[k],label="scatter")
        ax_ary[2][:set_xlabel](string("pred ",k))
        ax_ary[2][:set_ylabel](string("atual ",k))
    end
    ax_ary[1][:legend]()
    ax_ary[2][:legend]()
else
    fig,ax_ary = PyPlot.subplots(1,1,sharex=false)
    fig[:set_size_inches](8.0,6.0)

    scatter_with_conf_bnds(ax_ary,tst_in_eng,tst_out_eng_ary,:tprob,:X3_1,:X3_2,:red)

    ax_ary[:set_xlabel]("tprob")
    ax_ary[:set_ylabel](string(:X3_1))
    ax_ary[:axhline](limits[:X3_1][2])
    ax_ary[:axhline](limits[:X3_1][1])

    add_sq_annotation(ax_ary,tst_in_eng,tst_out_eng_ary,pred_outputs,50,:tprob,:X3_1,:X3_2,limits,[0.3,0.7])
    add_sq_annotation(ax_ary,tst_in_eng,tst_out_eng_ary,pred_outputs,200,:tprob,:X3_1,:X3_2,limits,[0.65,0.5])
    ax_ary[:text](0.5,limits[:X3_1][2],L"r_H",fontsize=15,va="bottom")
    ax_ary[:text](0.5,limits[:X3_1][1],L"r_L",fontsize=15,va="top")

    scatter_with_conf_bnds(ax_ary,tst_in_eng,pred_outputs,:tprob,:X3_1,:X3_2,:blue)
    #  ax_ary[:scatter](tst_in_eng[:tprob][x_srt],pred_outputs[:X3_1][x_srt],label="model",color=:blue)
    #  ucl = pred_outputs[:X3_1][x_srt]+pred_outputs[:X3_2][x_srt]
    #  lcl = pred_outputs[:X3_1][x_srt]-pred_outputs[:X3_2][x_srt]
    #  ax_ary[:scatter](tst_in_eng[:tprob][x_srt],ucl,label="model",s=0)
    #  ax_ary[:scatter](tst_in_eng[:tprob][x_srt],lcl,label="model",s=0)
    #  ax_ary[:fill_between](tst_in_eng[:tprob][x_srt],pred_outputs[:X3_1][x_srt],ucl,alpha=0.2,color=:blue)
    #  ax_ary[:fill_between](tst_in_eng[:tprob][x_srt],pred_outputs[:X3_1][x_srt],lcl,alpha=0.2,color=:blue)

end

#  PyPlot.legend()
show()
# 3d scatter
#  scatter3d(tst_in_eng[:tprob],tst_in_eng[:e_mcts],tst_out_eng,label="test",xlabel="tprob",ylabel="e",zlabel="exp rwd")
#  scatter3d!(tst_in_eng[:tprob],tst_in_eng[:e_mcts],pred_outputs[:X3_1])

#  plot(p1,p3,p4,p5,size=(1200,600))
#  plot(p1,size=(900,600))
