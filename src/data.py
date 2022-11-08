from dataclasses import dataclass
from keras_preprocessing.image import ImageDataGenerator, DirectoryIterator


@dataclass
class ImageData:
    """
    Class for wrapping dataset.

    This class generates a convenient wrapper for the image dataset we have.
    """

    training_data: DirectoryIterator
    validation_data: DirectoryIterator

    def __init__(self, datagen: ImageDataGenerator, dataset_path: str) -> None:
        """
        Init the ImageData generator class.

        Set the dataset path, and generate directory iterators for our dataset.
        """
        self.training_data = datagen.flow_from_directory(
            directory=dataset_path,
            target_size=(64, 64),
            batch_size=32,
            class_mode="binary",
            subset="training",
        )

        self.validation_data = datagen.flow_from_directory(
            directory=dataset_path,
            target_size=(64, 64),
            batch_size=32,
            class_mode="binary",
            subset="validation",
        )
