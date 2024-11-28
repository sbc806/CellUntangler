import ml_collections

def get_config():
    config = ml_collections.ConfigDict()

    config.seed = None
    config.device = "cpu"

    # Hyperparameters for dataset
    config.batch_size = 128
    config.dataset_size = None

    # Hyperparameters for model
    config.scalar_parametrization = False
    config.use_relu = False
    config.n_batch = [1]
    config.batch_invariant = True
    config.init = None
    config.gain = 1.0
    config.activation = "gelu"
    config.h_dim = 32
    config.use_hsic = False
    config.use_average_hsic = False
    config.hsic_weight = 1000

    # Hyperparmeters for training
    config.max_epochs = 500
    config.learning_rate = 0.001
    config.fixed_curvature = True
    # Betas
    config.start = 1.0
    config.end = 1.0
    config.end_epoch = 1
    config.epochs = 200
    # Batch normalization
    config.use_batch_norm = False

    # Hyperparameters for reconstruction
    config.reconstruction_term_weight = 1
    config.use_btcvae = False
    config.btcvae_beta = 1
    
    # Set the batch vector to be all zero if there is only one batch
    config.zero_batch = False
    config.print_batch = False

    return config
