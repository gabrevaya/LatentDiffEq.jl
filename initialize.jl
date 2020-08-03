
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
CuArrays.allowscalar(true)
using DiffEqGPU
using GPUArrays

# Flux needs to be in v0.11.0 (currently master, which is not compatible with DiffEqFlux compatibility, that's why I didn't include it in the Project.toml)

################################################################################
## Defining types

abstract type AbstractEncoder end
abstract type AbstractDecoder end
abstract type AbstractModel end

################################################################################
## Home files and modules

include("model_train.jl")
include("model_alt_train.jl")
include("utils/utils.jl")
include("utils/visualize.jl")
include("system/lv_problem.jl")

################################################################################
## Model definitions

include("model/model_manager.jl")
include("model/GOKU_model.jl")
include("model/latent_ode_model.jl")
