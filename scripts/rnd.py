import random
from pathlib import Path
import shutil


def main() -> None:
    image_dir = Path("./data/India")
    narrow_image_list = list(image_dir.iterdir())[3001:5001]

    rnd_lst = random.sample(narrow_image_list, 100)

    for path in rnd_lst:
        shutil.copy(path, "./data/random/")


if __name__ == "__main__":
    main()
