import random
from pathlib import Path
import shutil
import numpy as np


def main() -> None:
    image_names: list[str] = []

    image_dir = Path("./data/India")
    narrow_image_list = list(image_dir.iterdir())[5001:7001]

    rnd_lst = random.sample(narrow_image_list, 100)

    for path in rnd_lst:
        shutil.copy(path, "./data/random/")


if __name__ == "__main__":
    main()
