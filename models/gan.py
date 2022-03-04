import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import logging
import time
import logging
from tensorflow.keras.layers import GRU, Flatten, Dense, Conv1D, Dropout, LeakyReLU
from tensorflow.keras import Sequential
from sklearn.metrics import mean_squared_error
from pathlib import Path
import os
import sys


LOG_LEVEL = "INFO"


def prepare_logger():
    logger = logging.getLogger()
    logger.setLevel(level=LOG_LEVEL)
    formatter = logging.Formatter(
        "%(asctime)s,%(msecs)d %(levelname)-8s [%(pathname)s:%(lineno)d] %(message)s"
    )
    if not logger.handlers:
        lh = logging.StreamHandler(sys.stdout)
        lh.setFormatter(formatter)
        logger.addHandler(lh)
    return logger


logger = prepare_logger()
load_path = Path(os.path.abspath("")).parents[0] / "data" / "scaled_data"
names = ["X_list", "Y_preds_real_list", "Y_whole_real_list"]


def load_df_lists(names):
    data_dict = {}
    for name in names:
        with open(load_path / f"{name}.pickle", "rb") as handle:
            data_dict[name] = pickle.load(handle)


data_dict = load_df_lists(names)

with open(load_path / "X_list.pickle", "rb") as test:
    X_list = pickle.load(test)
with open(load_path / "Y_preds_real_list.pickle", "rb") as test:
    Y_preds_real_list = pickle.load(test)
with open(load_path / "Y_whole_real_list.pickle", "rb") as test:
    Y_whole_real_list = pickle.load(test)


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
            activation=LeakyReLU(alpha=0.01),
        )
    )
    model.add(
        Conv1D(
            64,
            kernel_size=5,
            strides=2,
            padding="same",
            activation=LeakyReLU(alpha=0.01),
        )
    )
    model.add(
        Conv1D(
            128,
            kernel_size=5,
            strides=2,
            padding="same",
            activation=LeakyReLU(alpha=0.01),
        )
    )
    model.add(Flatten())
    model.add(Dense(220, use_bias=False))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(220, use_bias=False, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model


class StockTimeGan:
    def __init__(self, generator, discriminator, learning_rate=0.00016):
        self.learning_rate = learning_rate
        self.generator = generator
        self.generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )
        self.discriminator = discriminator
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate
        )
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.checkpoint_directory = str(
            Path(os.path.abspath("")).parents[0] / "/checkpoints/"
        )
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
                generated_data, [generated_data.shape[0], generated_data.shape[1], 1]
            )
            real_to_pred_y_reshape = tf.reshape(
                Y_preds_real_list,
                [Y_preds_real_list.shape[0], Y_preds_real_list.shape[1], 1],
            )
            real_whole_y_reshape = tf.reshape(
                real_whole_y, [real_whole_y.shape[0], real_whole_y.shape[1], 1]
            )
            d_fake_input = tf.concat(
                [real_whole_y_reshape, tf.cast(generated_data_reshape, tf.float64)],
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
            zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )
        return real_to_pred_y, generated_data, disc_loss, gen_loss

    def train(self, real_x, real_to_pred_y, real_whole_y, epochs):
        train_history = {}
        train_history["gen_loss"] = []
        train_history["disc_loss"] = []
        train_history["real_y"] = []
        train_history["pred_y"] = []

        for i in range(epochs):
            start_time = time.time()
            real_to_pred_y, generated_data, disc_loss, gen_loss = self.train_step(
                real_x, real_to_pred_y, real_whole_y
            )
            train_history["gen_loss"].append(gen_loss)
            train_history["disc_loss"].append(disc_loss)
            train_history["real_y"].append(real_whole_y)
            train_history["pred_y"].append(generated_data)
            end_time = time.time()
            epoch_time = end_time - start_time
            rmse = np.sqrt(mean_squared_error(real_to_pred_y, generated_data))
            print(f"Epoch: {i} - RMSE: {rmse} - Epoch time: {epoch_time}")

        return train_history


if __name__ == "__main__":
    generator = generator(X_list.shape[1], X_list.shape[2])
    discriminator = discriminator((31, Y_preds_real_list.shape[1]))
    gan = StockTimeGan(generator, discriminator)
    train_history = gan.train(X_list, Y_preds_real_list, Y_whole_real_list, epochs=10)
