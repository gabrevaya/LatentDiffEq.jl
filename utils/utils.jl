
################################################################################
## Loss definitions

KL(μ, logσ²) = -logσ²/2f0 + ((exp(logσ²) + μ^2)/2f0) - 0.5f0

# make it better for gpu
CUDA.@cufunc KL(μ, logσ²) = -logσ²/2f0 + ((exp(logσ²) + μ^2)/2f0) - 0.5f0

# the calculation via log(var) = log(σ²) is more numerically efficient than through log(σ)
# KL(μ, logσ) = (exp(2f0 * logσ) + μ^2)/2f0 - 0.5f0 - logσ


function rec_loss(x, pred_x)

    # Data prep
    pred_x_stacked = Flux.stack(pred_x, 3)
    x_stacked = Flux.stack(x, 3)

    # Residual loss
    res = pred_x_stacked - x_stacked
    res_average = sum(mean((res).^2, dims = (2, 3)))

    # Differential residual loss
    res_diff = diff(pred_x_stacked, dims = 3) - diff(x_stacked, dims = 3)
    res_diff_average = sum(mean((res_diff).^2, dims = (2, 3)))

    return res_average + 1000*res_diff_average
end

function loss_batch(model::AbstractModel, λ, x, t, af)

    # Make prediction
    lat_var, pred_x, pred = model(x, t)

    # Compute reconstruction (and differential) loss
    reconstruction_loss = rec_loss(x, pred_x)

    # Compute KL losses from parameter and initial value estimation
    # kl_loss = 0
    # for i in 1:length(lat_var)
    #     μ, logσ² = lat_var[i]
    #     kl_loss += mean(sum(KL.(μ, logσ²), dims=1))
    # end

    # Filthy one liner that does the for loop above # lit
    kl_loss = sum( [ mean(sum(KL.(lat_var[i][1], lat_var[i][1]), dims=1)) for i in 1:length(lat_var) ] )

    return reconstruction_loss + af*(kl_loss)
end

## annealing factor parameters
# start_af: start value of annealing factor
# end_af: end value of annealing factor
# ae: annealing epochs - after these epochs, the annealing factor is set to the end value.
# epoch: current epoch
# mb_id: current number of mini batch
# mb_amount: amount of mini batches
function annealing_factor(start_af, end_af, ae, epoch, mb_id, mb_amount)

    if ae > 0
        if epoch < ae
            return start_af + (end_af - start_af) * (Float32(mb_id + epoch * mb_amount) / Float32(ae * mb_amount))
        else
            return end_af
        end
    else
        return 1.0
    end

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


################################################################################
## Training help function

function rand_time(full_seq_len, seq_len)
    start_time = rand(1:full_seq_len - seq_len)
    idxs = start_time:start_time+seq_len-1
    return idxs
end

function time_loader(x, full_seq_len, seq_len)

    x_ = Array{Float32, 3}(undef, (size(x,1), seq_len, size(x,3)))

    for i in 1:size(x,3)
        x_[:,:,i] = x[:,rand_time(full_seq_len, seq_len),i]
    end

    return Flux.unstack(x_, 2)

end

function create_prob(sys, u₀, tspan, p)

    generate_functions = ~(isfile("generated_f.jl") && isfile("generated_jac.jl") && isfile("generated_tgrad.jl"))

    if generate_functions
        computed_f = generate_function(sys, sparse = true)[2]
        computed_jac = generate_jacobian(sys, sparse = true)[2]
        computed_tgrad = generate_tgrad(sys, sparse = true)[2]

        open("generated_f.jl", "w") do io
         write(io, "f = $computed_f")
        end
        open("generated_jac.jl", "w") do io
         write(io, "jac = $computed_jac")
        end
        open("generated_tgrad.jl", "w") do io
         write(io, "tgrad = $computed_tgrad")
        end

        f = eval(computed_f)
        jac = eval(computed_jac)
        tgrad = eval(computed_tgrad)
    else
        include("generated_f.jl")
        include("generated_jac.jl")
        include("generated_tgrad.jl")
    end

    prob = ODEProblem(f, u₀, tspan, p, jac = jac, tgrad = tgrad)
    return prob
end
