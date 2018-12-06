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
                          output_dict::Dict=Dict(),nn_epoc::Int64=150,nn_batch_size::Int64=-1,
                          train_subsample::Int64=1,valid_subsample::Int64=1)

    training_in, training_out, training_table, input_sch, output_sch = return_data(train_fname,inputs=input_dict,outputs=output_dict,subsample=train_subsample)
    valid_in, valid_out, valid_table, _notused, _notused = return_data(valid_fname,inputs=input_dict,outputs=output_dict,subsample=valid_subsample)

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
    if nn_batch_size < 0
        # no value was provided
        nn_batch_size = round(Int,0.25*length(training_in))
    end

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

    # put into a SQ_model for return
    SQ = SQ_model(input_sch,output_sch,X3_1net,X3_2net,training_range)
    return SQ
end

function SQ_predict(SQ::SQ_model,inputs::Array;use_eng_units::Bool=false)
    println("inputs:")
    display(inputs)
    X3_1plotprovider = mx.ArrayDataProvider(:data => inputs, shuffle=false)
    X3_2plotprovider = mx.ArrayDataProvider(:data => inputs, shuffle=false)
    X3_1fit = mx.predict(SQ.X3_1net, X3_1plotprovider)
    X3_2fit = mx.predict(SQ.X3_2net, X3_2plotprovider)
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
        println(input_sch)
        display(in_data)
        display(output_sch)
        display(out_data)
        # this can happen from mis-spelled keys, OR data like a zero in the standard deviation
        error("NaN in data!!!")
    end

    return in_data, out_data, table, input_sch, output_sch
end

function add_sq_scatter3d_annotation(ax1::PyCall.PyObject,ax2::PyCall.PyObject,in::Array,out_eng_ary::Dict,i1::Symbol,i2::Symbol,poi::Array,SQmodel::SQ_model;yval::Symbol=:X3_1,yval_std::Symbol=:X3_2,marker::AbstractString="*",s::Int64=100,color=:black,fontsize::Int64=12)

    in_eng, pred_outputs = SQ_predict(SQmodel,in,use_eng_units=true)
    display(in_eng)

    x_idx = nearest_to_x(in,poi,val_and_idx=false)
    println("#####################")
    display(in_eng)
    println(size(in_eng[i1]))
    println(x_idx)
    println(poi)
    println("[$(in_eng[i1][x_idx]) $(in_eng[i2][x_idx])]")
    println("#####################")
    #  error()

    limits = restore_eng_units(SQmodel.range,SQmodel.output_sch)
    #  println(limits)
    rwd_rng = [limits[yval][1],limits[yval][2]]

    #  println(in_eng)
    #  println(pred_outputs)
    sq = X3(Normal(out_eng_ary[yval][x_idx],out_eng_ary[yval_std][x_idx]),Normal(pred_outputs[yval][x_idx],pred_outputs[yval_std][x_idx]),global_rwd_range=rwd_rng)
    sq_txt = @sprintf("%0.3f",sq)

    println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    println("vals: $([out_eng_ary[yval][x_idx], out_eng_ary[yval_std][x_idx]])")
    println("preds: $([pred_outputs[yval][x_idx], pred_outputs[yval_std][x_idx]])")
    println("pred_outputs: $pred_outputs")
    println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    ax2[:errorbar](0.1,pred_outputs[yval][x_idx],pred_outputs[yval_std][x_idx],color=:blue,label="Trusted",fmt="o",capsize=3,alpha=0.6)
    ax2[:errorbar](-0.1,out_eng_ary[yval][x_idx],out_eng_ary[yval_std][x_idx],color=:red,fmt="o",capsize=3,alpha=0.6,label="Candidate")
    ax2[:set_xlim]([-0.2,0.2])
    ax2[:set_xticklabels]("")
    #  ax2[:set_ylim](rwd_rng)
    ax2[:axhline](limits[:X3_1][2])
    ax2[:axhline](limits[:X3_1][1])
    ax2[:text](0.8*0.2,limits[:X3_1][2],L"r_H",fontsize=fontsize,va="top",ha="right")
    ax2[:text](0.8*0.2,limits[:X3_1][1],L"r_L",fontsize=fontsize,va="bottom",ha="right")
    ax2[:set_title](string("SQ at ",marker,": $sq_txt"),fontsize=fontsize)
    ax2[:set_ylabel]("Reward")
    ax2[:set_adjustable]("box-forced")
    #  ax2[:set_aspect](1/(5*(limits[:X3_1][2]-limits[:X3_2][1])))
end
function norm2(ary::Array)
    n = NaN*ones(1,size(ary,2))
    for i = 1:size(ary,2)
        n[i] = norm(ary[:,i])
    end
    return n
end
function nearest_to_x(ary::Array,x::Float64;val_and_idx::Bool=false)
    resid = abs.(ary-x)
    @assert size(resid,2) == size(ary,2)
    idx_min = indmin(resid)
    if val_and_idx
        return idx_min,ary[idx_min]
    else
        return idx_min
    end
end
function nearest_to_x(ary::Array,x::Array;val_and_idx::Bool=false)
    resid = norm2(ary.-x)
    idx_min = indmin(resid)
    println(idx_min)
    if val_and_idx
        return idx_min,ary[:,idx_min]
    else
        return idx_min
    end
end

function add_sq_annotation(ax::PyCall.PyObject,x_ary::Dict,y_ary::Dict,preds::Dict,idx::Int64,xvar::Symbol,yvar_mean::Symbol,yvar_std::Symbol,SQmodel::SQ_model;fontsize::Int64=12,txt_loc::Array=[])
    #  println("########## Debug Annotation ##########")
    #  idx, idx_val = nearest_to_x(tst_in_eng[xvar],target_val,val_and_idx=true)
    limits = restore_eng_units(SQmodel.range,SQmodel.output_sch)

    note_x = x_ary[xvar][idx]

    ex_c_m = y_ary[yvar_mean][idx]
    ex_c_s = y_ary[yvar_std][idx]
    ex_c = Normal(ex_c_m,ex_c_s)

    ex_t_m = preds[yvar_mean][idx]
    ex_t_s = preds[yvar_std][idx]
    ex_t = Normal(ex_t_m,ex_t_s)

    ex_X3 = X3(ex_c,ex_t,global_rwd_range=[limits[yvar_mean][1],limits[yvar_mean][2]])

    if isempty(txt_loc)
        # auto calculate the text location
        x_lims = (minimum(x_ary[xvar]),maximum(x_ary[xvar]))
        y_lims = (limits[yvar_mean][1],limits[yvar_mean][2])

        rng_xlims = x_lims[2]-x_lims[1]
        rng_ylims = y_lims[2]-y_lims[1]
        note_pct_x = (note_x-x_lims[1])/rng_xlims*1.1
        note_pct_y = (ex_c_m-y_lims[1])/rng_ylims*1.1
        txt_loc = [note_pct_x*rng_xlims+x_lims[1],note_pct_y*rng_ylims+y_lims[1]]
        #  println("##########")
        #  println("limits: $limits")
        #  println("nx: $note_x, xlims: $x_lims")
        #  println("ny: $ex_c_m, ylims: $y_lims")
        #  #  println("pct_coords: $")
        #  println("arrow coords: $([note_x,ex_c_m])")
        #  println("##########")
    end


    ax[:annotate](@sprintf("SQ:%0.2f",ex_X3),xy=[note_x,ex_c_m],xytext=txt_loc,arrowprops=Dict(:arrowstyle=>"->"),fontsize=fontsize)
    #  ax[:annotate](".",xy=[note_x,ex_c_m],xytext=[0.8,0.],arrowprops=Dict(:arrowstyle=>"->"),fontsize=fontsize)
    #  ax[:errorbar](note_x, ex_t_m,yerr=ex_t_s/2,fmt="o",capsize=3,color=:black,alpha=0.6,label="")
    #  println("########## END Debug Annotation ##########")
end

function return_cat_stats(y::Dict,yval::Symbol,x::Dict,xval::Symbol,yval_std::Symbol;return_std::Bool=false)
    unq_x = unique(x[xval])
    num_xs = length(unq_x)
    ymat = zeros(num_xs)
    t_ucl = zeros(num_xs)
    t_lcl = zeros(num_xs)
    std_val = NaN
    for u_x in unq_x
        u_x_ind = convert(Int,round(u_x))
        vals = y[yval][x[xval].==u_x]
        if length(unique(vals)) == 1
            #using model to predict
            ymat[u_x_ind] = mean(vals)
            std_val = y[yval_std][u_x_ind]/2
            t_ucl[u_x_ind] = ymat[u_x_ind]+std_val
            t_lcl[u_x_ind] = ymat[u_x_ind]-std_val
        else
            #  println(vals)
            ymat[u_x_ind] = mean(vals)
            #  t_ucl[u_x_ind] = ymat[u_x_ind]+y[yval_std][u_x_ind]/2
            #  t_lcl[u_x_ind] = ymat[u_x_ind]-y[yval_std][u_x_ind]/2
            std_val = std(vals)
            t_ucl[u_x_ind] = ymat[u_x_ind]+std_val
            t_lcl[u_x_ind] = ymat[u_x_ind]-std_val
        end
    end
    if return_std
        return ymat,unq_x,std_val
    else
        return ymat,unq_x,t_ucl,t_lcl
    end
end

function scatter_with_conf_bnds(ax::PyCall.PyObject,x::Dict,y::Dict,xval::Symbol,yval::Symbol,yval_std::Symbol,color::Symbol;subsample::Array{Int64}=collect(1:length(x[xval])),label::String="",bar::Bool=false)
    unq_x = unique(x[xval])
    num_xs = length(unq_x)
    if bar
        x_srt2 = sortperm(x[xval])[subsample]
        ax[:errorbar](x[xval][subsample], y[yval][subsample],yerr=y[yval_std][subsample],fmt="o",capsize=3,color=color,alpha=0.6,label=label)
    else
        x_srt = sortperm(x[xval])[subsample]

        ax[:plot](x[xval][x_srt],y[yval][x_srt],label=label,color=color)
        t_ucl = y[yval]+y[yval_std]
        t_lcl = y[yval]-y[yval_std]
        #  println(t_ucl)
        #  println(x_srt)

        ax[:scatter](x[xval][x_srt],t_ucl[x_srt],label="",s=0)
        ax[:scatter](x[xval][x_srt],t_lcl[x_srt],label="",s=0)

        ax[:fill_between](x[xval][x_srt],y[yval][x_srt],t_ucl[x_srt],alpha=0.2,color=color)
        ax[:fill_between](x[xval][x_srt],y[yval][x_srt],t_lcl[x_srt],alpha=0.2,color=color)
    end

end
function make_label_from_keys(d::Dict)
    z = ""
    for x in collect(keys(d))
        z = string(z,x)
    end
    return z
end
function make_label_from_keys(a::Array)
    d = Dict()
    for z in a
        d[z] = ""
    end
    return make_label_from_keys(d)
end

function make_sq_model(net_type::String,inpts::Array{Symbol};num_epoc::Int64=250,subsample::Int64=1,
                       trusted_fname::String="",net_folder::String="nn_logs")
    train_fname = trusted_fname
    test_fname = trusted_fname #see if this works... I don't this is important for training

    #  inputs = Dict(:tprob=>"ML.Continuous",:e_mcts=>"ML.Continuous")
    inputs = Dict()
    for i in inpts
        inputs[i] = "ML.Continuous"
    end
    outputs = Dict(:X3_1=>"ML.Continuous",:X3_2=>"ML.Continuous")

    log_fname = joinpath(net_folder,"$(net_type)_$(make_label_from_keys(inputs))")

    training_subsample = subsample
    # make model
    SQmodel = make_nn_SQ_model(train_fname,test_fname,log_fname,input_dict=inputs,output_dict=outputs,nn_epoc=num_epoc,train_subsample=training_subsample)

    info("Writing variable to file:")
    println(log_fname)
    jldopen(log_fname*"_SQmodel.jld","w") do file
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
end
