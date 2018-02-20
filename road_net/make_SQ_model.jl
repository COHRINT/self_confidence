using JuliaDB
using MXNet
using Plots

#  train_table = loadtable("logs/transition_vary.csv",escapechar='"')
#  train_table = loadtable("logs/transition_vary_2.csv",escapechar='"')
train_table = loadtable("logs/transition_vary_4.csv",escapechar='"')
#  train_table = loadtable("logs/net_vary.csv",escapechar='"')
bad_table = loadtable("logs/transition_vary_test_bad_solver.csv",escapechar='"')
ok_table = loadtable("logs/transition_vary_test_ok_solver.csv",escapechar='"')

#  test_table = loadtable("logs/transition_vary_test.csv",escapechar='"')
#  test_table = loadtable("logs/transition_vary.csv",escapechar='"')
#  test_table = loadtable("logs/transition_vary_test_2.csv",escapechar='"')
test_table = loadtable("logs/transition_vary_test_4.csv",escapechar='"')
#  test_table = loadtable("logs/net_vary_test.csv",escapechar='"')

fig_name = "net_plot.png"

# use "nothing" to ignore certain variables...for now
sch = ML.schema(train_table, hints=Dict(
                :graphID=>nothing,
                :discount=>nothing,
                #  :discount=>ML.Continuous,
                :tprob=>ML.Continuous,
                #  :tprob=>nothing,
                :num_exit_nodes=>nothing,
                :exit_rwd=>nothing,
                :caught_rwd=>nothing,
                :sensor_rwd=>nothing,
                #  :avg_degree=>ML.Continuous,
                :avg_degree=>nothing,
                :deg_variance=>nothing,
                #  :deg_variance=>ML.Continuous,
                :diam=>nothing,
                :max_degree=>nothing,
                :N=>nothing,
                :E=>nothing,
                :its=>nothing,
                :e_mcts=>nothing,
                :d_mcts=>nothing,
                :steps=>nothing,
                :repeats=>nothing,
                :exit_distance=>nothing,
                :pursuer_distance=>nothing,
                #  :expected_rwd=>nothing,
                :expected_rwd=>ML.Continuous,
                #  :upm_lpm=>ML.Continuous,
                :upm_lpm=>nothing,
                :mean=>nothing,
                :median=>nothing,
                :moment_2=>nothing,
                :moment_3=>nothing,
                :moment_4=>nothing,
                :moment_5=>nothing,
                :moment_6=>nothing,
                :moment_7=>nothing,
                :moment_8=>nothing,
                :moment_9=>nothing,
                :moment_10=>nothing,
                ))

function my_splitschema(xs::ML.Schema,ks::Vector{Symbol})
    return filter((k,v) -> k ∉ ks, xs), filter((k,v) -> k ∈ ks, xs)
end

tprob = [z.tprob for z in train_table]
exp_rwd = [z.expected_rwd for z in train_table]
upm_lpm = [z.upm_lpm for z in train_table]

#  output_list = [:expected_rwd, :upm_lpm]
#  output_list = [:upm_lpm]
#  input_list = [:deg_variance]
input_list = [:tprob]
output_list = [:expected_rwd]
input_sch, output_sch = my_splitschema(sch,output_list)

training_in = ML.featuremat(input_sch, train_table)
training_out = ML.featuremat(output_sch, train_table)
#  training_in = NDArray(tprob')
#  training_out = NDArray([exp_rwd upm_lpm]')

tst_exp_rwd = [z.expected_rwd for z in test_table]
tst_upm_lpm = [z.upm_lpm for z in test_table]
tst_tprob = [z.tprob for z in test_table]

bad_exp_rwd = [z.expected_rwd for z in bad_table]
bad_tprob = [z.tprob for z in bad_table]
ok_exp_rwd = [z.expected_rwd for z in ok_table]
ok_tprob = [z.tprob for z in ok_table]

#use input and output_sch already obtained to transform test data in the same way training data was transformed
test_input = ML.featuremat(input_sch, test_table)
test_output = ML.featuremat(output_sch, test_table)
#  test_input = NDArray(tst_tprob')
#  test_output = NDArray([tst_exp_rwd tst_upm_lpm]')


function data_source(batchsize::Int64)
  train = mx.ArrayDataProvider(
    :data => training_in,
    :label => training_out,
    batch_size = batchsize,
    shuffle = true,
    )
  valid = mx.ArrayDataProvider(
    :data => test_input,
    :label => test_output,
    batch_size = batchsize,
    shuffle = true,
    )
  train, valid
end

data = mx.Variable(:data)
label = mx.Variable(:label)
net = @mx.chain     mx.Variable(:data) =>
                    mx.FullyConnected(num_hidden=6) =>
                    mx.Activation(act_type=:relu) =>
                    mx.FullyConnected(num_hidden=6) =>
                    mx.Activation(act_type=:relu) =>
                    mx.FullyConnected(num_hidden=6) =>
                    mx.Activation(act_type=:relu) =>
                    mx.FullyConnected(num_hidden=6) =>
                    mx.Activation(act_type=:relu) =>
                    mx.FullyConnected(num_hidden= length(output_list)) =>
                    mx.LinearRegressionOutput(mx.Variable(:label))

# final model definition, don't change, except if using gpu
model = mx.FeedForward(net, context=mx.cpu())

# set up the optimizer: select one, explore parameters, if desired
#  optimizer = mx.SGD(η=0.01, μ=0.9, λ=0.00001)
optimizer = mx.ADAM()

# train, reporting loss for training and evaluation sets
# initial training with small batch size, to get to a good neighborhood
trainprovider, evalprovider = data_source(#= batchsize =# 25)
mx.fit(model, optimizer, trainprovider,
       initializer = mx.NormalInitializer(0.0, 0.1),
       eval_metric = mx.MSE(),
       eval_data = evalprovider,
       n_epoch = 1.500e3,
       callbacks = [mx.speedometer()])

# obtain predictions
plotprovider = mx.ArrayDataProvider(:data => test_input, :label => test_output, shuffle=false)
fit = mx.predict(model, plotprovider)

# convert data back to original units
function restore_eng_units(ary::Array,sch::ML.Schema)
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
    return [x.second for x in arrays]
end

fit_eng = restore_eng_units(fit,output_sch)

#  println(fit_eng)
tst_out_eng = restore_eng_units(test_output,output_sch)
tst_in_eng = restore_eng_units(test_input,input_sch)
input_eng = restore_eng_units(test_input,input_sch)
#  input_eng = input_eng[:tprob]
#  input_eng = input_eng[:deg_variance]

println("correlation between fitted values and true regression line: ", cor(vec(fit), vec(test_output)))
text_labels = ["reward" "upm_lpm"]
if length(output_list) == 1
    output_var = output_list[1]
    input_var = input_list[1]

    input_string = string(input_var)
    output_string = string(output_var)

    x1 = tst_in_eng[input_var]
    x2 = tst_out_eng[output_var]
    y = fit_eng[output_var]

    y_range = (minimum(y),maximum(y))
    x1_range = (minimum(x1),maximum(x1))
    x2_range = (minimum(x2),maximum(x2))

    p1 = Plots.scatter(x1,y,label="SQ NN Model, D=8",color=:black)
    #  p1 = Plots.scatter(tst_tprob,tst_exp_rwd,label=output_string)
    scatter!(ok_tprob,ok_exp_rwd,label="D=3",color=:blue,m=:+)
    title!("$output_string vs. $input_string")
    xaxis!("$input_string",x1_range)
    yaxis!("$output_string")

    p2 = Plots.scatter(x1,y,label="SQ NN Model, D=8",color=:black)
    #  p1 = Plots.scatter(tst_tprob,tst_exp_rwd,label=output_string)
    scatter!(bad_tprob,bad_exp_rwd,label="D=1",color=:green,marker=:star4,markersize=5,markerstrokewidth=0)
    title!("$output_string vs. $input_string")
    xaxis!("$input_string",x1_range)
    yaxis!("$output_string")

    Plots.plot(p1,p2,size=(1200,500))
else
    p1 = Plots.scatter(tst_in_eng[:avg_degree],fit_eng[:expected_rwd],label="expected_reward")
    p2 = Plots.scatter(tst_in_eng[:avg_degree],fit_eng[:upm_lpm],label="upm_lpm")
    Plots.plot(p1,p2)
    Plots.savefig(fig_name,dpi=300)
    Plots.show()
end
#  Plots.scatter(input_eng,fit_eng[:upm_lpm],label=text_labels,layout=2)
