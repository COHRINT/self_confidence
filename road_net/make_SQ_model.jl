using JuliaDB
using MXNet
using Plots
using ProgressMeter

mutable struct SQ_model
    input_sch::ML.Schema
    output_sch::ML.Schema
    network #neural network
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
  train, valid
end

function make_nn_SQ_model(train_fname::String,valid_fname::String; input_dict::Dict=Dict(),
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
    mx.fit(network, optimizer, trainprovider,
           initializer = mx.NormalInitializer(0.0, 0.1),
           eval_metric = mx.MSE(),
           eval_data = evalprovider,
           n_epoch = nn_epoc,
           callbacks = [mx.speedometer()])

    # put into a SQ_model for return
    SQ = SQ_model(input_sch,output_sch,network)
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
    for entry in sch
        i = 1
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

    if isempty(schema)
        schema_dict = Dict()
        println("##### getting data #####")
        #  p = Progress(length(colnames(table)),dt=0.1,desc="Processing",output=STDOUT)
        for itm_name in colnames(table)
            if itm_name ∈ keys(inputs)
                #  ProgressMeter.next!(p,showvalues=[itm_name])
                schema_dict[itm_name] = inputs[itm_name]
            elseif itm_name ∈ keys(outputs)
                #  ProgressMeter.next!(p,showvalues=[itm_name])
                schema_dict[itm_name] = outputs[itm_name]
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

# problem setup
#  train_fname = "logs/transition_vary_4.csv"
#  test_fname = "logs/transition_vary_test_4.csv"
train_fname = "logs/transition_e_vary_reference_solver_training.csv"
test_fname = "logs/transition_e_vary_ok_solver.csv"
inputs = Dict(:tprob=>ML.Continuous)
#  inputs = Dict(:tprob=>ML.Continuous,:e_mcts=>ML.Continuous)
outputs = Dict(:expected_rwd=>ML.Continuous)

# make model
SQmodel = make_nn_SQ_model(train_fname,test_fname,input_dict=inputs,output_dict=outputs,nn_epoc=200,nn_batch_size=50)

# get test and make predictions
test_input, test_output, test_table, input_sch, output_sch = return_data(test_fname, inputs=inputs, outputs=outputs)
_notused, pred_outputs = SQ_predict(SQmodel,test_input,test_output,use_eng_units=true)

tst_out_eng = restore_eng_units(test_output,output_sch)[:expected_rwd]

#plot results
p1 = Plots.scatter(test_input',tst_out_eng,title="tprob vs rwd",label="true")
scatter!(test_input',pred_outputs[:expected_rwd],label="model")
#  p2 = Plots.scatter(pred_inputs[:e_mcts],test_output',title="e_mcts vs rwd",label="")
#  p2 = Plots.scatter(test_input[2,:],test_output',title="e_mcts vs rwd",label="")
p3 = Plots.scatter(pred_outputs[:expected_rwd],tst_out_eng,title="pred vs actual",label="")
#  p4 = Plots.scatter3d(repmat(test_input[1,:],750,1),repmat(test_input[2,:],750,1),repmat(test_output',750,1))
#  p4 = Plots.scatter3d(test_input[1,:]',test_input[2,:]',test_output)

#  plot(p1,p2,p3,p4,size=(1200,600))
plot(p1,p3,size=(1200,600))
