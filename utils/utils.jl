
################################################################################
## Loss definitions

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
    #     kl_loss += mean(sum(kl.(μ, logσ²), dims=1))
    # end

    kl_loss = sum( [ mean(sum(kl.(μ[i], logσ²[i]), dims=1)) for i in 1:length(μ) ] )
    
    return reconstruction_loss + kl_loss
end

kl(μ, logσ²) = -logσ²/2f0 + ((exp(logσ²) + μ^2)/2f0) - 0.5f0

# make it better for gpu
# CUDA.@cufunc kl(μ, logσ²) = -logσ²/2f0 + ((exp(logσ²) + μ^2)/2f0) - 0.5f0

function rec_loss(x, x̂)

    # Data prep
    x̂_stacked = Flux.stack(x̂, 3)
    x_stacked = Flux.stack(x, 3)

    # Residual loss
    res = x̂_stacked - x_stacked
    res_average = sum(mean((res).^2, dims = (2, 3)))

    # Differential residual loss
    # res_diff = diff(x̂_stacked, dims = 3) - diff(x_stacked, dims = 3)
    # res_diff_average = sum(mean((res_diff).^2, dims = (2, 3)))

    # return (res_average + 1000f0*res_diff_average)/size(pred_x_stacked,1)
    return res_average/size(x̂_stacked,1)
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

function normalize_to_unit_segment(X)
    min_val = minimum(X)
    max_val = maximum(X)

    X̂ = (X .- min_val) ./ (max_val - min_val)
    return X̂, min_val, max_val
end

denormalize_unit_segment(X̂, min_val, max_val) = X̂ .* (max_val .- min_val) .+ min_val


################################################################################
## Training help function

function time_loader(x, full_seq_len, seq_len)

    x_ = Array{Float32, 3}(undef, (size(x,1), seq_len, size(x,3)))

    for i in 1:size(x,3)
        x_[:,:,i] = x[:,rand_time(full_seq_len, seq_len),i]
    end

    return Flux.unstack(x_, 2)

end

################################################################################
## Visualization functions

function visualize_val_image(model, val_set, t_val, h, w)
    j = rand(1:size(val_set,3))
    X_test = val_set[:,:,j]
    frames_test = [Gray.(reshape(x,h,w)) for x in eachcol(X_test)]
    X_test = reshape(X_test, Val(3))
    x = Flux.unstack(X_test, 2)

    X̂, μ, logσ² = model(x, t_val)
    x̂, ẑ, ẑ₀, = X̂

    if length(X̂) == 4
        θ̂ = X̂[4]
        @show θ̂
    end

    # gr(size = (700, 350))
    ẑ = Flux.stack(ẑ, 2)

    plt = plot(ẑ[1,:,1], legend = false)
    ylabel!("Angle")
    xlabel!("time")
    # plt = plot(ẑ[1,1,:]) # for Latent ODE
    display(plt)

    x̂ = Flux.stack(x̂, 2)
    frames_pred = [Gray.(reshape(x,h,w)) for x in eachslice(x̂, dims=2)]

    frames_test = frames_test[1:6:end]
    frames_pred = frames_pred[1:6:end]

    plt = mosaicview(frames_test..., frames_pred..., nrow=2, rowmajor=true)
    display(plt)
end


function visualize_val_image(model, val_set, t_val, h, w, color_scheme)
    j = rand(1:size(val_set,3))
    X_test = val_set[:,:,j]
    frames_test = [get.(Ref(color_scheme), reshape(x,h,w)) for x in eachcol(X_test)]
    X_test = reshape(X_test, Val(3))
    x = Flux.unstack(X_test, 2)

    X̂, μ, logσ² = model(x, t_val)
    x̂, ẑ, ẑ₀, θ̂ = X̂
    x̂ = Flux.stack(x̂, 2)

    if !isnan(x̂[1])
        frames_pred = [get.(Ref(color_scheme), reshape(x,h,w)) for x in eachslice(x̂, dims=2)]

        frames_test = frames_test[1:6:end]
        frames_pred = frames_pred[1:6:end]
        
        plt = mosaicview(frames_test..., frames_pred..., nrow=2, rowmajor=true)
        display(plt)
    end
end