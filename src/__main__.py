import logging
import os
import re
from typing import Optional
from setup import setup
from keras_preprocessing.image import ImageDataGenerator
import data
import model
from keras import models


def build_model_from_scratch(img_data: data.ImageData) -> model.NeuralNet:
    local_logger = logging.getLogger("tracker")

    # Setup the neural network and train it
    cnn = model.NeuralNet()

    cnn.train(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
        data=img_data,
        epochs=25,
    )

    local_logger.info("Neural network tranining done")

    return cnn


def find_saved_model() -> Optional[str]:
    local_logger = logging.getLogger("tracker")

    pattern = re.compile("^cnn.*\.h5$")
    for filepath in os.listdir("./"):
        if pattern.match(filepath):
            local_logger.info("Found saved model: %s", filepath)
            return filepath

    local_logger.info("No saved model found.")
    return None


def main() -> None:
    local_logger = logging.getLogger("tracker")

    # Basic setup
    setup()

    local_logger.info("Logger setup finished.")

    # Load data from directory
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,
    )

    img_data = data.ImageData(datagen, "./data/final_data")
    local_logger.info("Data loaded.")

    # Load the model from a file if it exists or train it
    cnn = None

    saved_model = find_saved_model()

    if saved_model is None:
        cnn = build_model_from_scratch(img_data)
    else:
        network = models.load_model(saved_model)
        cnn = model.NeuralNet(network=network)

    # Save the model for good meausure
    cnn.network.save("cnn_anomaly.h5")

    local_logger.info(
        "Normal image predication: %s", cnn.pred_image("./data/test/normal.jpg")
    )

    local_logger.info(
        "Pothole image prediction: %s", cnn.pred_image("./data/test/negative.jpg")
    )

if __name__ == "__main__":
    main()
