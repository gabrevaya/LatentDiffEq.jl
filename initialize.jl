
## To run before using the code

################################################################################
## Julia packages

using OrdinaryDiffEq
using Base.Iterators: partition
using BSON:@save, @load
using BSON
using CUDA
using DrWatson: struct2dict
using DiffEqFlux
using Flux
using Flux.Data: DataLoader
import Flux.Data: _getobs
using Flux: reset!
using FileIO
using JLD2
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
using DiffEqGPU
CUDA.allowscalar(false)

################################################################################
## Defining types

abstract type AbstractEncoder end
abstract type AbstractDecoder end
abstract type AbstractModel end
abstract type AbstractSystem end

################################################################################
## Home files and modules

include("model_train.jl")
include("model_alt_train.jl")
include("utils/utils.jl")
include("utils/visualize.jl")
include("system/Lotka-Volterra.jl")
include("system/van_der_Pol.jl")
include("system/Wilson-Cowan.jl")


################################################################################
## Model definitions

include("model/model_manager.jl")
include("model/GOKU_model.jl")
include("model/latent_ode_model.jl")
