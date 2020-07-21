
include("GOKU_train.jl")

function lv_func(du, u, p, t)
      x, y = u
      α, β, δ, γ = p
      @inbounds begin
            du[1] = α*x - β*x*y
            du[2] = -δ*y + γ*x*y
      end
      nothing
end

function generate_dataset(; kws...)

      args = Args(; kws...)

      ##########################################################################
      ## Problem definition

      prob = ODEProblem(lv_func, [0., 0.], args.full_t_span, [0., 0., 0., 0.], jac=true, sparse=true)

      ##########################################################################
      ## Function definition

      output_func(sol, i) = (Array(sol), false)
      rand_uniform(range::Tuple, size) = rand(Uniform(range...),(size,1))
      rand_uniform(range::Array, size) = [rand(Uniform(r[1], r[2])) for r in eachrow(range)]

      # for samping initial condition from a uniform distribution
      function prob_func(prob,i,repeat)
            u0_new = rand_uniform(args.u₀_range, length(prob.u0))
            prob = remake(prob, u0 = u0_new)
      end

      gen = lv_gen(args.ode_dim, args.hidden_dim_gen, args.input_dim)

      ##########################################################################
      ## Create data

      # generate some fixed random parameters
      p = rand_uniform(args.p₀_range, length(prob.p))
      prob = remake(prob, p = p)

      # Solve for X trajectories
      ensemble_prob = EnsembleProblem(prob, prob_func=prob_func, output_func = output_func)
      sim = solve(ensemble_prob, Tsit5(), saveat=args.dt, trajectories=10000)

      raw_data = dropdims(Array(sim), dims = 2)
      
      # Probably works but requieres alot of RAM for some reason
      # norm_data = zeros(Float32, size(raw_data))
      # norm_data = normalize_Z(raw_data)

      gen_data = gen.linear.(Flux.unstack(raw_data, 2))
      gen_data = Flux.stack(gen_data, 2)

      @save args.data_file_name raw_data gen_data p gen
end

function solve_prob(u0, p, tspan, tstep)

      @parameters t α β δ γ
      @variables x(t) y(t)
      @derivatives D'~t

      u₀ = [x => u0[1]
            y => u0[2]]
      p = [ α => p[1]
            β => p[2]
            δ => p[3]
            γ => p[4]]

      eqs = [D(x) ~ α*x - β*x*y,
           D(y) ~ -δ*y + γ*x*y]

      sys = ODESystem(eqs)
      prob = ODEProblem(sys, u₀, tspan, p, jac=true, sparse=true)
      sol = solve(prob, Vern7(), saveat = tstep)

      sol
end


################################################################################
## Data pre-processing

# normalize raw data passed as a 3D array (input_dim, time, trajectories)
function normalize_Z(data)

      data = Flux.unstack(data, 3)

      μ = 0.
      σ = 0.
      for i in 1:size(data,1)
            for j in 1:size(data[1],1)
                  μ = mean(data[i][j,:])
                  σ = std(data[i][j,:])
                  data[i][j,:] = ( data[i][j,:] .- μ ) ./ σ
            end
      end

      data = Flux.stack(data, 3)

      return norm_data

end
