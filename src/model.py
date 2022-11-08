import logging
from typing import Optional
from keras import models, layers
from keras_preprocessing import image
import data
import numpy as np


class NeuralNet:
    def __init__(self, network: Optional[models.Sequential] = None):
        if network is not None:
            self.network = network
        else:
            self.network = models.Sequential(
                layers=[
                    layers.Conv2D(
                        filters=32,
                        kernel_size=3,
                        activation="relu",
                        input_shape=[64, 64, 3],
                    ),
                    # Hidden layers till first Dense layer
                    layers.MaxPool2D(pool_size=2, strides=2),
                    layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
                    layers.MaxPool2D(pool_size=2, strides=2),
                    layers.Flatten(),
                    # Hidden layer with complete connections
                    layers.Dense(units=128, activation="relu"),
                    # Output layer
                    layers.Dense(units=1, activation="sigmoid"),
                ]
            )

    def train(
        self,
        optimizer: str,
        loss: str,
        metrics: list[str],
        data: data.ImageData,
        epochs: int,
    ) -> None:
        """
        Compile and fit the network according to the given data class.
        """

        local_logger = logging.getLogger("tracker")
        local_logger.info("Compiling network with the following parameters: ")
        local_logger.info(f"optimizer: {optimizer}, loss: {loss}, metrics: {metrics}")

        self.network.compile(optimizer, loss, metrics)

        local_logger.info(f"Training model. No. of epochs: {epochs}")

        self.network.fit(
            x=data.training_data, validation_data=data.validation_data, epochs=epochs
        )

    def pred_image(self, img_path: str) -> str:
        """
        Function to predict the class of a single image using the trained network.
        """
        img = image.load_img(img_path, target_size=(64, 64))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        result = self.network(img)

        return "pothole" if result[0][0] == 1 else "normal"
