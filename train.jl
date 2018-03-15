using Flux
using Flux: onehotbatch, argmax, crossentropy, throttle, @epochs
using Base.Iterators: repeated, partition
using MLDatasets

BATCH_SIZE = 100

# load CIFAR-10 training set
trainX, trainY = CIFAR10.traindata()
testX,  testY  = CIFAR10.testdata()

# MLDatasets returns UInt8 thus convert it to Float64
trainX = Array{Float64}(trainX)
testX = Array{Float64}(testX)
println("conversion is done")

# construct one-hot vectors from labels
trainY = onehotbatch(trainY, 0:9)
testY = onehotbatch(testY, 0:9)

train = (trainX, trainY)


# TODO convert below to list comprehension
# TODO shuffle
# split training set into batches
# train_data contains whole data in batches
train_data = Array{Any}(div(50000, BATCH_SIZE))
for i = 0:div(50000, BATCH_SIZE) - 1
    train_data[i+1] = train[1][:,:,:, 1 + i*BATCH_SIZE:(i+1)*BATCH_SIZE],
                      train[2][:, 1 + i*BATCH_SIZE:(i+1)*BATCH_SIZE]
end

println("data is ready to be learnt")

m = Chain(
  Conv((3,3), 3=>16, relu),
  x -> maxpool(x, (2,2)),
  Conv((2,2), 16=>8, relu),
  x -> maxpool(x, (2,2)),
  x -> reshape(x, :, size(x, 4)),
  Dense(8*7*7 , 10), softmax)

m(train_data[1][1])
loss(x, y) = crossentropy(m(x), y)
accuracy(x, y) = mean(argmax(m(x)) .== argmax(y))
evalcb = throttle(() -> @show(accuracy(testX, testY)), 10)
opt = ADAM(params(m))

# Flux.train!() runs for 1 epoch, default. 
# Change 15 to train for different epochs using @epochs macro
@epochs 15 Flux.train!(loss, train, opt, cb = evalcb)
