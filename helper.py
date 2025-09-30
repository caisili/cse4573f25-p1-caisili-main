'''
Helper and zip functions.
Please read the instructions before you start task2.

Please do NOT make any change to this file.
'''

import os, torch
import zipfile, argparse
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F


def show_image(img: torch.Tensor, delay=1000):
    """Shows an image.
    """
    plt.imshow(F.to_pil_image(img))
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="CSE 473/573 project Geometry submission.")
    parser.add_argument("--ubit", type=str)
    args = parser.parse_args()
    return args


def files2zip(files: list, zip_file_name: str):
    with zipfile.ZipFile(zip_file_name, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            path, name = os.path.split(file)
            if os.path.exists(file):
                zf.write(file, arcname=name)


if __name__ == "__main__":
    args = parse_args()
    file_list = ['geometry.py', 'result_task1.json', 'result_task2.json']
    files2zip(file_list, 'submission_' + args.ubit + '.zip')
