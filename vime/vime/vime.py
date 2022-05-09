import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout


class VIME_Self(tf.keras.models.Model):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features

        self.encoder = Sequential(
            [
                Dense(self.num_features, activation="relu"),
            ]
        )

        self.feature_decoder = Sequential(
            [
                Dense(self.num_features, activation="sigmoid"),
            ]
        )

        self.mask_estimator = Sequential(
            [
                Dense(self.num_features, activation="sigmoid"),
            ]
        )

    def call(self, x):
        # Reconstruct features and mask from corrupted data
        encoded = self.encoder(x)
        x_reconstruct = self.feature_decoder(encoded)
        mask_reconstruct = self.mask_estimator(encoded)

        return {"mask": mask_reconstruct, "feature": x_reconstruct}


class VIME(tf.keras.models.Model):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

        self.predictor = Sequential(
            [
                Dense(128, activation="relu"),
                Dense(128, activation="relu"),
                Dense(1, activation='sigmoid'),
            ]
        )

    def call(self, x):
        x_encoded = self.encoder(x)
        y = self.predictor(x_encoded)
        
        return y