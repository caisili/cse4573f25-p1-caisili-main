'''
Camera Calibration: 
Please read the instructions before you start task2.

Please do NOT make any change to this file.
'''
import json, torch
import torchvision.io as io
from helper import show_image
from geometry import find_intrinsic, find_extrinsic
from geometry import find_corner_img_coord, find_corner_world_coord


def save_result(
    img_coord: torch.Tensor, 
    world_coord: torch.Tensor,
    fx, fy, cx, cy: float,
    R, T: torch.Tensor,
    save_path='result_task2.json',
):
    result = {}
    result['img_coord'] = img_coord.tolist()
    result['world_coord'] = world_coord.tolist()
    result['fx'] = fx
    result['fy'] = fy
    result['cx'] = cx
    result['cy'] = cy
    result['R'] = R.tolist()
    result['T'] = T.tolist()
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":

    img = io.read_image('checkboard.png', mode=io.ImageReadMode.GRAY)
    img_coord = find_corner_img_coord(img)
    world_coord = find_corner_world_coord(img_coord)
    fx, fy, cx, cy = find_intrinsic(img_coord, world_coord)
    R, T = find_extrinsic(img_coord, world_coord)
    save_result(img_coord, world_coord, fx, fy, cx, cy, R, T)

    print('img_coord:')
    print(img_coord)
    print('world_coord:')
    print(world_coord)
    print('fx, fy, cx, cy')
    print(fx, fy, cx, cy)
    print('R')
    print(R)
    print('T')
    print(T)
    # show_image(img)
