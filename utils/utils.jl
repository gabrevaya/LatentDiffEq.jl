
################################################################################
## Loss definitions

KL(μ, logσ²) = -logσ²/2f0 + ((exp(logσ²) + μ^2)/2f0) - 0.5f0

# make it better for gpu
# CUDA.@cufunc KL(μ, logσ²) = -logσ²/2f0 + ((exp(logσ²) + μ^2)/2f0) - 0.5f0

# the calculation via log(var) = log(σ²) is more numerically efficient than through log(σ)
# KL(μ, logσ) = (exp(2f0 * logσ) + μ^2)/2f0 - 0.5f0 - logσ


function rec_loss(x, x̂)

    # Data prep
    x̂_stacked = Flux.stack(x̂, 3)
    x_stacked = Flux.stack(x, 3)

    # Residual loss
    res = x̂_stacked - x_stacked
    # res_av1 = mean((res).^2, dims = 2)
    # @show typeof(res_av1)
    # res_av1[:,:,1] .*= 10.f0
    # res_average = sum(mean(res_av1, dims = 3))

    res_average = sum(mean((res).^2, dims = (2, 3)))

    # Differential residual loss
    # res_diff = diff(pred_x_stacked, dims = 3) - diff(x_stacked, dims = 3)
    # res_diff_average = sum(mean((res_diff).^2, dims = (2, 3)))

    # return (res_average + 1000f0*res_diff_average)/size(pred_x_stacked,1)
    return res_average/size(x̂_stacked,1)
end

function rec_ini_loss(x, pred)
    # Data prep
    x_stacked = Flux.stack(x, 3)
    res = mean((pred[1] - x_stacked[:,:,1]).^2)
end

# think how to dispatch this for GOKU related only and not any AbstractModel
function loss_batch(model::LatentDiffEqModel, λ, x, t, af)

    # Make prediction
    X̂, μ, logσ² = model(x, t)
 
    x̂, ẑ, ẑ₀, = X̂

    # Compute reconstruction (and differential) loss
    reconstruction_loss = rec_loss(x, x̂)
    # rec_initial_condition_loss = rec_ini_loss(x, pred)
    # @show rec_initial_condition_loss

    # Compute KL losses from parameter and initial value estimation
    # kl_loss = 0
    # for i in 1:length(lat_var)
    #     μ, logσ² = lat_var[i]
    #     kl_loss += mean(sum(KL.(μ, logσ²), dims=1))
    # end

    kl_loss = sum( [ mean(sum(KL.(μ[i], logσ²[i]), dims=1)) for i in 1:length(μ) ] )
    
    return reconstruction_loss + kl_loss
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
            return Float32(start_af + (end_af - start_af) * (mb_id + epoch * mb_amount) / (ae * mb_amount))
        else
            return Float32(end_af)
        end
    else
        return 1.0f0
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


function NormalizeToUnitSegment(X)
    min_val = minimum(X)
    max_val = maximum(X)

    X̂ = (X .- min_val) ./ (max_val - min_val)
    return X̂, min_val, max_val
end

DenormalizeUnitSegment(X̂, min_val, max_val) = X̂ .* (max_val .- min_val) .+ min_val


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

function time_loader2(x, full_seq_len, seq_len)

    x_ = Array{Float32, 3}(undef, (size(x,1), seq_len, size(x,3)))

    for i in 1:size(x,3)
        x_[:,:,i] = x[:,rand_time(full_seq_len, seq_len),i]
    end

    x_samples_unstacked = Flux.unstack(x_, 3)
    return x_samples_unstacked

end

function create_prob(sys_name, k, sys, u₀, tspan, p)

    func_folder = mkpath(joinpath("precomputed_systems", sys_name))
    osc_folder = mkpath(joinpath(func_folder, "oscillators_"*string(k)))
    # f_file = joinpath(osc_folder, "generated_f.jld2")
    # jac_file = joinpath(osc_folder, "generated_jac.jld2")
    # tgrad_file = joinpath(osc_folder, "generated_tgrad.jld2")
    f_file = joinpath(osc_folder, "generated_f.bson")
    jac_file = joinpath(osc_folder, "generated_jac.bson")
    tgrad_file = joinpath(osc_folder, "generated_tgrad.bson")

    generate_functions = ~(isfile(f_file) && isfile(jac_file) && isfile(tgrad_file))

    if generate_functions
        computed_f = generate_function(sys, sparse = true)[2]
        computed_jac = generate_jacobian(sys, sparse = true)[2]
        computed_tgrad = generate_tgrad(sys, sparse = true)[2]

        f = eval(computed_f)
        jac = eval(computed_jac)
        tgrad = eval(computed_tgrad)

        @save(f_file, f)
        @save(jac_file, jac)
        @save(tgrad_file, tgrad)
        
    else
        @info "Precomputed functions found for the system"
        @load(f_file, f)
        @load(jac_file, jac)
        @load(tgrad_file, tgrad)

        # print(f)
    end
    prob = ODEProblem(f, u₀, tspan, p, jac = jac, tgrad = tgrad)
    return prob
end



## for ILC


# function rec_loss(x::Array{T,2}, pred_x) where T
#     pred_stacked = Flux.stack(pred_x, 2)
#     # Residual loss
#     res = x - pred_stacked
#     res_average = mean((res).^2, dims = (1, 2))
#     return res_average[1]
# end

# function ILC_train(x, model, λ, t, af, device, ILC_threshold, ps)
#     grads = Zygote.Grads[]
#     for sample in x
#         loss, back = Flux.pullback(ps) do
#             # Compute loss
#             loss_batch(model, λ, sample |> device, t, af)
#         end
#         # Backpropagate
#         grad = back(1f0)
#         push!(grads, grad)
#     end

#     masking!.(ps, Ref(grads), ILC_threshold)
#     return grads[1]
# end

# function masking!(p, grads, threshold)
#     if ~(grads[1][p] == nothing)
#         mean_signs = mean([sign.(el[p]) for el in grads])
#         mask = mean_signs .< threshold
#         mean_grads = mean([el[p] for el in grads])
#         mean_grads[mask] .= 0.f0
#         grads[1][p][:] = mean_grads[:]
#     end
# end

# rec_ini_loss(x::Array{T,2}, pred) where T = mean((pred[1] - x[:,1]).^2)
