
## To run before using the code

################################################################################
## Julia packages

using OrdinaryDiffEq
using Base.Iterators: partition
using BSON:@save, @load
using BSON
using CUDAapi: has_cuda_gpu ## TODO: use CUDA package instead (device()s)
using DrWatson: struct2dict
using DiffEqFlux
using Flux
using Flux.Data: DataLoader
import Flux.Data: _getobs
using Flux: reset!
using Logging: with_logger
using Parameters: @with_kw
using ProgressMeter: Progress, next!
using Random
using MLDataUtils
using Statistics
using Zygote
using Plots
using Distributions
using ModelingToolkit
using CuArrays
CuArrays.allowscalar(false)

# Flux needs to be in v0.11.0 (currently master, which is not compatible with DiffEqFlux compatibility, that's why I didn't include it in the Project.toml)

################################################################################
## Home files and modules

include("utils/utils.jl")
include("utils/visualize.jl")
include("model/model_manager.jl")
include("system/lv_problem.jl")
include("model_train.jl")
include("model/GOKU_model.jl")
include("model/latent_ode_model.jl")

using .GOKU_model
using .Latent_ODE_model

################################################################################
## Arguments for the train function
@with_kw mutable struct Args

    ## Model and problem definition
    model_name = "GOKU"
    problem = "lv"

    ## Training params
    η = 1e-3                    # learning rate
    λ = 0.01f0                  # regularization paramater
    batch_size = 256            # minibatch size
    seq_len = 100               # sampling size for output
    epochs = 200                # number of epochs for training
    seed = 1                    # random seed
    cuda = false                # GPU usage
    dt = 0.05                   # timestep for ode solve
    t_span = (0.f0, 4.95f0)     # span of time interval for training
    start_af = 0.00001          # Annealing factor start value
    end_af = 0.00001            # Annealing factor end value
    ae = 200                    # Annealing factor epoch end

    ## Progressive observation training
    progressive_training = false    # progressive training usage
    obs_seg_num = 6             # number of step to progressive training
    start_seq_len = 100         # training sequence length at first step
    full_seq_len = 400          # training sequence length at last step

    ## Model dimensions
    rnn_input_dim = 32          # rnn input dimension
    rnn_output_dim = 32         # rnn output dimension
    latent_dim = 4              # latent dimension
    hidden_dim = 120            # hidden dimension
    hidden_dim_node = 200       # hidden dimension of the neuralODE
    hidden_dim_gen = 10         # hidden dimension of the g function

    ## Save paths and keys
    save_path = "output"        # results path
    data_file_name = "lv_data.bson"  # data file name
    raw_data_name = "raw_data"  # raw data name
    gen_data_name = "gen_data"  # generated data name

end
