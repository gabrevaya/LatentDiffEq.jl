# Example of GOKU-net model on friction-less pendulum data created with Luxor

using LatentDiffEq
using FileIO
using Random
using Statistics
using MLDataUtils
using BSON: @save
using Flux.Data: DataLoader
using Flux
using OrdinaryDiffEq
# using StochasticDiffEq
using ModelingToolkit
using DiffEqSensitivity
using Flux.Optimise: update!
using FastAI
using FastAI:FluxTraining
# using Images
# using Plots
# import GR
using CUDA
CUDA.allowscalar(false)

include("pendulum.jl")
include("create_data.jl")

################################################################################
## Arguments for the train function
## Global model
model_type = GOKU()

## Latent Differential Equations
diffeq = Pendulum()

## Training params
η = 1e-3                        # learning rate
decay = 0.001f0                 # decay applied to weights during optimisation
batch_size = 64                 # minibatch size
seq_len = 50                    # sequence length for training samples
epochs = 10                   # number of epochs for training
seed = 333                      # random seed
cuda = true                     # GPU usage (not working well yet)
dt = 0.05                       # timestep for ode solve
variational = true              # variational or deterministic training

## Annealing schedule
start_β = 0.00001f0             # start value
end_β = 0.00001f0               # end value
n_cycle = 4                     # number of annealing cycles
ratio = 0.9                     # proportion used to increase β (and 1-ratio used to fix β)

## Progressive observation training
progressive_training = false    # progressive training usage
prog_training_duration = 200    # number of epochs to reach the final seq_len
start_seq_len = 10              # training sequence length at first step

## Visualization
vis_len = 60                    # number of test frames to visualize after each epoch
save_figure = true              # true: save visualization figure in save_path folder
                                # false: display image instead of saving it    

################################################################################
################################################################################
## Training done manualy

# if cuda && has_cuda_gpu()
#     device = gpu
#     @info "Training on GPU"
# else
#     device = cpu
#     @info "Training on CPU"
# end
device = cpu

############################################################################
## Prepare training data
root_dir = @__DIR__
data_path = "$root_dir/data/data.bson"

if ~isfile(data_path)
    @info "Generating data"
    latent_data, u0s, ps, high_dim_data = generate_dataset(diffeq = diffeq)
    data = (latent_data, u0s, ps, high_dim_data)
    mkpath("$root_dir/data")
    @save data_path data
end

seed > 0 && Random.seed!(seed)

data_loaded = load(data_path, :data)
train_data = data_loaded[4]
latent_data = data_loaded[1]
params_data = data_loaded[3]

# Stack time for each sample
train_data = Flux.stack.(train_data, 3)

# Stack all samples
train_data = Flux.stack(train_data, 4) # 28x28x400x450
h, w, full_seq_len, observations = size(train_data)
latent_data = Flux.stack(latent_data, 3)
params_data = Flux.stack(params_data, 3)

# Vectorize frames
train_data = reshape(train_data, :, full_seq_len, observations) # input_dim, time_size, samples
train_data = Float32.(train_data)

# Split into train and validation sets
train_set, val_set = Array.(splitobs(train_data, 0.9))
train_set_latent, val_set_latent = Array.(splitobs(latent_data, 0.9))
train_set_params, val_set_params = Array.(splitobs(params_data, 0.9))

# Prepare data loader
loader_train = DataLoader(train_set, batchsize=batch_size, shuffle=true, partial=false)

val_set = permutedims(val_set, [1,3,2])
t_val = range(0.f0, step=dt, length = size(val_set, 3))
input_dim = size(train_set, 1)

############################################################################
# Create model
encoder_layers, decoder_layers = default_layers(model_type, input_dim, diffeq, device = device)
model = LatentDiffEqModel(model_type, encoder_layers, decoder_layers)

############################################################################
## Define optimizer
opt = ADAM(η)
# opt = AdaBelief(η)
#opt = ADAMW(η, (0.9,0.999), decay)

function loss(x, x̂, μ, logσ², β = 1)
    reconstruction_loss = sum(mean((x .- x̂).^2, dims=(2,3)))
    kl_loss = vector_kl(μ, logσ²)
    return reconstruction_loss + β * kl_loss
end

learner = Learner(model, (training = loader_train,), opt, loss)

struct VAETrainingPhase <: FluxTraining.AbstractTrainingPhase end

function FluxTraining.step!(learner, phase::VAETrainingPhase, batch)
    FluxTraining.runstep(learner, phase, (x = batch,)) do handle, state
        ps = learner.params
        # x = gpu(reduce(hcat, state.x))
        # Permute dimesions for having (pixels, batch_size, time)
        x = PermutedDimsArray(state.x, [1,3,2])
        # Use only random sequences of length seq_len for the current minibatch
        x = time_loader(x, full_seq_len, seq_len) |> device       
        t = range(0.f0, step=dt, length=seq_len)

        state.grads = gradient(ps) do
            # get encode, sample latent space, decode
            X̂, μ, logσ² = learner.model(x, t, true)
            x̂, ẑ, l̂ = X̂

            handle(FluxTraining.LossBegin())
            state.loss = learner.lossfn(x, x̂, μ, logσ²)

            handle(FluxTraining.BackwardBegin())
            return state.loss
        end
        handle(FluxTraining.BackwardEnd())
        update!(learner.optimizer, ps, state.grads)
    end
end

for epoch in 1:10
    epoch!(learner, VAETrainingPhase())
end