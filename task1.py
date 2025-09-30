'''
Camera Calibration: 
Please read the instructions before you start task1.

Please do NOT make any change to this file.
'''
import json, argparse, torch
from geometry import findRot_xyz2XYZ, findRot_XYZ2xyz


def parse_args():
    parser = argparse.ArgumentParser(description="CSE 473/573 project Geometry.")
    parser.add_argument("--alpha", type=float, default=45)
    parser.add_argument("--beta",  type=float, default=30)
    parser.add_argument("--gamma", type=float, default=50)
    args = parser.parse_args()
    return args


def save_result(rot_xyz2XYZ: torch.Tensor, rot_XYZ2xyz: torch.Tensor, save_path='result_task1.json'):
    result = {}
    result['rot_xyz2XYZ'] = rot_xyz2XYZ.tolist()
    result['rot_XYZ2xyz'] = rot_XYZ2xyz.tolist()
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":

    args = parse_args()
    rot_xyz2XYZ = findRot_xyz2XYZ(args.alpha, args.beta, args.gamma)
    rot_XYZ2xyz = findRot_XYZ2xyz(args.alpha, args.beta, args.gamma)
    save_result(rot_xyz2XYZ, rot_XYZ2xyz)

    print('rot_xyz2XYZ:')
    print(rot_xyz2XYZ)
    print('rot_XYZ2xyz:')
    print(rot_XYZ2xyz)
