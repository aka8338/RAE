import tensorflow as tf
from tensorflow.keras import layers, regularizers

class RegularizedAutoencoder(tf.keras.Model):  # Class for a regularized autoencoder model
    def __init__(self, input_dim=784, hidden_dims=[512, 256, 128], latent_dim=20, weight_decay=1e-4):
        super(RegularizedAutoencoder, self).__init__()
        
        # Encoder: A sequential model that compresses the input into a latent representation
        self.encoder = tf.keras.Sequential([
            layers.Dense(hidden_dims[0], activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Dense(hidden_dims[1], activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Dense(hidden_dims[2], activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Dense(latent_dim)
        ])
        
        # Decoder: A sequential model that reconstructs the input from the latent representation
        self.decoder = tf.keras.Sequential([
            layers.Dense(hidden_dims[2], activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Dense(hidden_dims[1], activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Dense(hidden_dims[0], activation='relu', kernel_regularizer=regularizers.l2(weight_decay)),
            layers.Dense(input_dim, activation='sigmoid')
        ])

    def call(self, inputs):
        z = self.encoder(inputs)  # Encode the input to latent representation
        reconstructed = self.decoder(z)  # Decode the latent representation back to input space
        return reconstructed  # Return the reconstructed input
