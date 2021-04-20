# LatentDiffEq
Latent Differential Equation models in Julia.

#### Note: This repo is still experimental!

## Installation

1. Clone the repository and enter the folder:
```
$ git clone git@github.com:gabrevaya/LatentDE.git
$ cd LatentDE
```

2. Download the training data from [this link](https://drive.google.com/file/d/1Td7zvvFk5An9DqcaCXAPjeKDqc05dB_R/view?usp=sharing).

3. Extract the archive, and move its content following this path “pendulum_friction-less/data/“

4. Instantiate and try the GOKU-net model on a pendulum example
```julia
$ julia
julia> ] activate .
julia> ] instantiate
julia> include("LatentDiffEq.jl")
julia> include("pendulum_friction-less/model_train.jl")
julia> train()
```

## References

Linial, O., Eytan, D., & Shalit, U. (2020). Generative ODE Modeling with Known Unknowns. arXiv preprint [arXiv:2003.10775](https://arxiv.org/abs/2003.10775).
