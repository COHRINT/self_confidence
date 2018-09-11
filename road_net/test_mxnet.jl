using MXNet
using Distributions
using Plots

# create training and evaluation data sets
samplesize  = 1000
TrainInput = rand(1,samplesize)*10
TrainOutput = sin.(TrainInput)+rand(1,samplesize)*0.5
ValidationInput = rand(1,samplesize)*10
ValidationOutput = sin.(ValidationInput)+rand(1,samplesize)*0.5

error()

# how to set up data providers using data in memory
function data_source(batchsize = 100)
  train = mx.ArrayDataProvider(
    :data => TrainInput,
    :label => TrainOutput,
    batch_size = batchsize,
    shuffle = true,
    )
  valid = mx.ArrayDataProvider(
    :data => ValidationInput,
    :label => ValidationOutput,
    batch_size = batchsize,
    shuffle = true,
    )

  train, valid
end

# create a two hidden layer MPL: try varying num_hidden, and change tanh to relu,
# or add/remove a layer
data = mx.Variable(:data)
label = mx.Variable(:label)
net = @mx.chain     mx.Variable(:data) =>
                    mx.FullyConnected(num_hidden=6) =>
                    mx.Activation(act_type=:tanh) =>
                    #  mx.FullyConnected(num_hidden=6) =>
                    #  mx.Activation(act_type=:tanh) =>
                    #  mx.FullyConnected(num_hidden=6) =>
                    #  mx.Activation(act_type=:tanh) =>
                    #  mx.FullyConnected(num_hidden=6) =>
                    #  mx.Activation(act_type=:tanh) =>
                    mx.FullyConnected(num_hidden=1) =>
                    mx.LinearRegressionOutput(mx.Variable(:label))

# final model definition, don't change, except if using gpu
#  model = mx.FeedForward(net, context=mx.cpu())
model = mx.FeedForward(net, context=mx.gpu())

# set up the optimizer: select one, explore parameters, if desired
#optimizer = mx.SGD(η=0.01, μ=0.9, λ=0.00001)
optimizer = mx.ADAM()

# more training with the full sample
trainprovider, evalprovider = data_source(#= batchsize =# 200)
mx.fit(model, optimizer, trainprovider,
       initializer = mx.NormalInitializer(0.0, 0.1),
       eval_metric = mx.MSE(),
       eval_data = evalprovider,
       n_epoch = 10,  # previous setting is batchsize = 200, epoch = 20
                       # implies we did (5000 / 200) * 20 times update in previous `fit`
       callbacks = [mx.speedometer()])

# obtain predictions
plotprovider = mx.ArrayDataProvider(:data => ValidationInput, :label => ValidationOutput)
fit = mx.predict(model, plotprovider)
println("correlation between fitted values and true regression line: ", cor(vec(fit), vec(ValidationOutput)))
scatter(ValidationOutput',fit',w = 3, xlabel="true", ylabel="predicted", title="45º line is what we hope for", show=true)
