import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    # Hyperparameters for dataset
    config.batch_size = 128
    
    # Hyperparameters for model
    config.scalar_parametrization = False
    config.use_relu = False
    config.n_batch = [1]
    config.batch_invariant = True
    config.init = None
    config.activation = "gelu"
    config.use_hsic = False
    config.hsic_weight = 1000

    # Hyperparmeters for training
    config.max_epochs = 500
    config.learning_rate = 0.001
    config.fixed_curvature = True

    return config
