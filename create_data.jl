using ModelingToolkit
# using DifferentialEquations
using OrdinaryDiffEq
using Random
using Distributions

@parameters t α β δ γ
@variables x(t) y(t)
@derivatives D'~t

eqs = [D(x) ~ α*x - β*x*y,
       D(y) ~ -δ*y + γ*x*y]

sys = ODESystem(eqs)

 u0 = [x => 1.0
       y => 1.0]

 p  = [α => 1.5
       β => 1.0
       δ => 3.0
       γ => 1.0]

 tspan = (0.0,19.95)
 prob = ODEProblem(sys,u0, tspan, p, jac=true, sparse=true)
 sol = solve(prob, Vern7(), saveat = 0.1)

output_func(sol,i) = (Array(sol),false)

p₀ = [pᵢ[2] for pᵢ ∈ p]
u₀ = [u₀ᵢ[2] for u₀ᵢ ∈ u0]
# u₀_range = [0.2 2.0
#             0.2 2.0]

u₀_range = (1.5, 3.0) # Goku-nets uses: init_state = np.random.uniform(1.5, 3.0, size=self.state_size)
p₀_range = (1.0, 2.0) # Goku-nets uses: rand_params = np.random.uniform(1.0, 2.0, size=4)

rand_uniform(range::Tuple, size) = rand(Uniform(range...),(size,1))
rand_uniform(range::Array, size) = [rand(Uniform(r[1], r[2])) for r in eachrow(range)]

function prob_func(prob,i,repeat)
      u0_new = rand_uniform(u₀_range, length(prob.u0))
      p_new = rand_uniform(p₀_range, length(prob.p))
      remake(prob, u0 = u0_new, p = p_new)
end

function prob_func2(prob,i,repeat)
      p_new = p₀ + 0.1*randn(length(p₀))
      u0_new  = u₀ + 0.1*randn(length(u₀))
      remake(prob, u0 = u0_new, p = p_new)
end

ensemble_prob = EnsembleProblem(prob, prob_func=prob_func, output_func = output_func)
# ensemble_prob = EnsembleProblem(prob, prob_func=prob_func)
sim = solve(ensemble_prob, Vern7(), saveat=0.05, trajectories=10000)

full_data = dropdims(Array(sim), dims = 2)

# when not passing the output_func, we can plot it like this
# using Plots
# plot(sim,linealpha=0.2, color=:blue, vars=(1))
# plot!(sim,linealpha=0.2, color=:red, vars=(2))


using BSON: @save, @load

@save "lv_data.bson" full_data

# @load "lv_data.bson" full_data
#
# using BSON
# BSON.parse("lv_data.bson")


# Goku-net splits the 10000 trajectories into 90% for training and 10% for testing
# It also saves the parameters corresponding to each trajectory. If we see it is
# necessary, we can also save them, includding them in the output_func.

# They also uses an emsission function:
# self.net = nn.Sequential(
#     nn.Linear(state_dim, hidden_dim),
#     nn.ReLU(),
#     nn.Linear(hidden_dim, sample_dim)
# )
# with hidden_dim = 10 and sample_dim = 4
#
# We can start without it and later add it.
