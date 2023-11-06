using Flux, CSV, DataFrames, Plots, JLD2

# Runtime Configuration
newModel = false;

# Data Loading -> data = (Samples, Features)
data = CSV.File("data.csv", header=1) |> DataFrame;

# Data Preprocessing
features = data[:, 2:end];
labels = data[:, 1];
EncodedLabels = Flux.onehotbatch(labels, 0:9);
classes = 10; # [0, 9]

# Data Splitting
train, val, test = Flux.MLUtils.splitobs((features, EncodedLabels), at=(0.6, 0.2));

# Data Normalization
nTrain = Flux.MLUtils.normalise(Matrix(Flux.MLUtils.getobs(train[1])));
nTest = Flux.MLUtils.normalise(Matrix(Flux.MLUtils.getobs(test[1])));
nVal = Flux.MLUtils.normalise(Matrix(Flux.MLUtils.getobs(val[1])));

lTrain = Matrix(Flux.MLUtils.getobs(train[2]));
lTest = Matrix(Flux.MLUtils.getobs(test[2]));
lVal = Matrix(Flux.MLUtils.getobs(val[2]));

# Hyperparameters
epochs = 1000;
α = 0.1; # Learning Rate
ψ = 0.0001; # Momentum
λ = 0.0004; # Weight Decay
batchSize = 250;

# Batching
batches = Flux.DataLoader((nTrain', lTrain[:, :]), batchsize = batchSize, shuffle = true);

# Network Architecture
inputNodes = length(features[1, :]);
hiddenNodes = [25, 25, 25];
outputNodes = classes;

# Network Initialization
if !newModel
    model = JLD2.load("Model.jld2")["model"];
else
    model = Chain(
        Dense(inputNodes, hiddenNodes[1], Flux.relu , init=Flux.glorot_uniform),
        Dense(hiddenNodes[1], hiddenNodes[2], Flux.relu, init=Flux.glorot_uniform),
        Dense(hiddenNodes[2], hiddenNodes[3], Flux.relu, init=Flux.glorot_uniform),
        Dense(hiddenNodes[3], outputNodes, Flux.relu, init=Flux.glorot_uniform),
        softmax
    );
end

# Loss Function
loss(x, y) = Flux.crossentropy(model(x), y);

# Optimizer
opt = Flux.Optimiser(Flux.WeightDecay(λ), Flux.Momentum(α, ψ));

# Metrics
# [Training, Validation]
lossLog = zeros(epochs, 2);
accuracyLog = zeros(epochs, 2);

# Training
for epoch in 1:epochs
    for miniBatch in batches
        Flux.train!(loss, Flux.params(model), [miniBatch], opt);
    end

    lossLog[epoch, 1] = loss(nTrain', lTrain[:, :]);
    lossLog[epoch, 2] = loss(nVal', lVal[:, :]);
    accuracyLog[epoch, 1] = (sum(Flux.OneHotArrays.onecold(model(nTrain')) .== Flux.OneHotArrays.onecold(lTrain)) / length(lTrain))*1000;
    accuracyLog[epoch, 2] = (sum(Flux.OneHotArrays.onecold(model(nVal')) .== Flux.OneHotArrays.onecold(lVal)) / length(lVal))*1000;
    
    if epoch % 1 == 0
        println("Epoch: ", epoch, " | Training Loss: ", lossLog[epoch, 1], " | Training Accuracy: ", accuracyLog[epoch, 1], "%", " | Validation Loss: ", lossLog[epoch, 2], " | Validation Accuracy: ", accuracyLog[epoch, 2], "%");
    end
end
TestAccuracy = (sum(Flux.OneHotArrays.onecold(model(nTest')) .== Flux.OneHotArrays.onecold(lTest)) / length(lTest))*1000;
println("Test Accuracy: ", TestAccuracy, "%");

# Save Model
JLD2.jldsave("Model.jld2"; model);

# Loss Visualization
plot(1:epochs, lossLog[:, 1], label="Training Loss", xlabel="Epochs", ylabel="Loss", title="Loss");
display(plot!(1:epochs, lossLog[:, 2], label="Validation Loss", xlabel="Epochs", ylabel="Loss"));
plot(1:epochs, accuracyLog[:, 1], label="Training Accuracy", xlabel="Epochs", ylabel="Accuracy", title="Accuracy");
display(plot!(1:epochs, accuracyLog[:, 2], label="Validation Accuracy", xlabel="Epochs", ylabel="Accuracy"));