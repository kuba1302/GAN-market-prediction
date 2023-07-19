import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

from utils.log import prepare_logger

cuda_path = Path(
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin"
)
os.add_dll_directory(str(cuda_path))

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Conv1D, Dense, Flatten, LeakyReLU

LOG_LEVEL = "INFO"
logger = prepare_logger(LOG_LEVEL)


def generator(input_dim, feature_size, output_dim=1):
    model = Sequential()
    model.add(
        GRU(
            units=1024,
            return_sequences=True,
            input_shape=(input_dim, feature_size),
            recurrent_dropout=0.2,
        )
    )
    model.add(GRU(units=512, return_sequences=True, recurrent_dropout=0.2))
    model.add(GRU(units=256, return_sequences=False, recurrent_dropout=0.2))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(units=output_dim))
    return model


def discriminator(input_shape):
    model = tf.keras.Sequential()
    model.add(
        Conv1D(
            32,
            input_shape=input_shape,
            kernel_size=3,
            strides=2,
            padding="same",
            activation=LeakyReLU(alpha=0.1),
        )
    )
    model.add(
        Conv1D(
            64,
            kernel_size=5,
            strides=2,
            padding="same",
            activation=LeakyReLU(alpha=0.1),
        )
    )
    model.add(
        Conv1D(
            128,
            kernel_size=5,
            strides=2,
            padding="same",
            activation=LeakyReLU(alpha=0.1),
        )
    )
    model.add(Flatten())
    model.add(Dense(220, use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(220, use_bias=False, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model


class StockTimeGan:
    def __init__(
        self,
        generator,
        discriminator,
        checkpoint_directory,
        learning_rate=0.0016,
    ):
        self.learning_rate = learning_rate
        self.generator = generator
        self.generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )
        self.discriminator = discriminator
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.checkpoint_directory = checkpoint_directory
        self.checkpoint_prefix = str(Path(self.checkpoint_directory) / "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator,
        )

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.loss(tf.ones_like(real_output), real_output)
        fake_loss = self.loss(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.loss(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, real_x, real_to_pred_y, real_whole_y):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = self.generator(real_x, training=True)
            generated_data_reshape = tf.reshape(
                generated_data,
                [generated_data.shape[0], generated_data.shape[1], 1],
            )
            real_to_pred_y_reshape = tf.reshape(
                real_to_pred_y,
                [real_to_pred_y.shape[0], real_to_pred_y.shape[1], 1],
            )
            real_whole_y_reshape = tf.reshape(
                real_whole_y, [real_whole_y.shape[0], real_whole_y.shape[1], 1]
            )
            d_fake_input = tf.concat(
                [
                    real_whole_y_reshape,
                    tf.cast(generated_data_reshape, tf.float64),
                ],
                axis=1,
            )
            d_real_input = tf.concat(
                [real_whole_y_reshape, real_to_pred_y_reshape], axis=1
            )
            real_output = self.discriminator(d_real_input, training=True)
            fake_output = self.discriminator(d_fake_input, training=True)
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, self.discriminator.trainable_variables
        )

        self.generator_optimizer.apply_gradients(
            zip(gradients_of_generator, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(
                gradients_of_discriminator,
                self.discriminator.trainable_variables,
            )
        )
        return real_to_pred_y, generated_data, disc_loss, gen_loss

    def train(self, real_x, real_to_pred_y, real_whole_y, epochs):
        train_history = {}
        train_history["gen_loss"] = []
        train_history["disc_loss"] = []
        train_history["real_y"] = []
        train_history["pred_y"] = []

        for i in tqdm(range(epochs), desc="GAN TRAINING EPOCHS"):
            start_time = time.time()
            realy_y, generated_data, disc_loss, gen_loss = self.train_step(
                real_x, real_to_pred_y, real_whole_y
            )
            train_history["gen_loss"].append(gen_loss.numpy())
            train_history["disc_loss"].append(disc_loss.numpy())
            train_history["real_y"].append(real_whole_y)
            train_history["pred_y"].append(generated_data.numpy())
            end_time = time.time()
            epoch_time = end_time - start_time
            rmse = np.sqrt(mean_squared_error(realy_y, generated_data))
            mae = mean_absolute_error(y_true=realy_y, y_pred=generated_data)
            logger.info(
                f"Epoch: {i + 1} - RMSE: {rmse} MAE : {mae} - Epoch time: {epoch_time} - Discriminator Loss: {disc_loss} - Generator Loss: {gen_loss}"
            )
            if epochs % 10:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

        return train_history

    def predict(self, X, *args, **kwargs):
        return self.generator.predict(X, args, kwargs)

    def save_generator(self, save_path):
        self.generator.save(save_path)
