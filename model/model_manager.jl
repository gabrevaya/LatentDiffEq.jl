
################################################################################
## Model manager in order to create right model from those available and assign it the right physical system definition
function initialize_model(args, device)

    @unpack_Args args       # unpack all arguments
    input_dim = Int(0)      # model input size
    ode_dim = Int(0)        # ode solve size
    p_dim = Int(0)          # number of parameter of system

    if problem == "lv"

        ## Problem defined dimension definition
        input_dim = 2
        ode_dim = 2
        p_dim = 4

        ## ODE function
        # func = lv_func    # No jacobian definition
        func = ODEFunction(lv_func, jac = lv_jac, tgrad = lv_tgrad)

    end

    if model_name == "GOKU"

        # Create model
        model = Goku(input_dim, latent_dim, rnn_input_dim, rnn_output_dim, hidden_dim, ode_dim, p_dim, func, Tsit5(), device)

        # Get parameters
        ps = Flux.params(model.encoder.linear, model.encoder.rnn, model.encoder.rnn_μ, model.encoder.rnn_logσ², model.encoder.lstm, model.encoder.lstm_μ, model.encoder.lstm_logσ², model.decoder.z₀_linear, model.decoder.p_linear, model.decoder.gen_linear)

    elseif model_name == "latent_ode"

        # Create model
        model = Latent_ODE(input_dim, latent_dim, hidden_dim, rnn_input_dim, rnn_output_dim, hidden_dim_node, seq_len, t_span[2], device)

        # Get parameters
        ps = Flux.params(model.encoder.linear, model.encoder.rnn, model.encoder.μ, model.encoder.logσ²,
                         model.decoder.dudt, model.decoder.linear)

    end

    model, ps
end
