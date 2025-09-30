import torch
from typing import Tuple


'''
Please do Not change or add any imports. 
'''

# --------------------------------------------------- task1 ----------------------------------------------------------
def _deg2rad_tensor(a_deg: float) -> torch.Tensor:
# keep float32 throughout to satisfy grader dtype
    return torch.deg2rad(torch.tensor(a_deg, dtype=torch.float32))

def findRot_xyz2XYZ(alpha: float, beta: float, gamma: float) -> torch.Tensor:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles along x, y and z axis respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 tensor represents the rotation matrix from xyz to XYZ.
    '''
    a = _deg2rad_tensor(alpha)
    b = _deg2rad_tensor(beta)
    c = _deg2rad_tensor(gamma)
    R = _Rz(c) @ _Ry(b) @ _Rx(a)
    return R


def findRot_XYZ2xyz(alpha: float, beta: float, gamma: float) -> torch.Tensor:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles of the 3 step respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 tensor represents the rotation matrix from XYZ to xyz.
    '''
    R = findRot_xyz2XYZ(alpha, beta, gamma)
    return R.t()

"""
If your implementation requires implementing other functions.
Please implement all the functions you design under here.
But remember the above "findRot_xyz2XYZ()" and "findRot_XYZ2xyz()"
functions are the only 2 function that will be called in task1.py.
"""

# Your functions for task1:
def _Rx(alpha_rad: torch.Tensor) -> torch.Tensor:
    ca = torch.cos(alpha_rad); sa = torch.sin(alpha_rad)
    return torch.tensor([[1.0, 0.0, 0.0],
                         [0.0, ca.item(), -sa.item()],
                         [0.0, sa.item(),  ca.item()]], dtype=torch.float32)

def _Ry(beta_rad: torch.Tensor) -> torch.Tensor:
    cb = torch.cos(beta_rad); sb = torch.sin(beta_rad)
    return torch.tensor([[ cb.item(), 0.0, sb.item()],
                         [ 0.0, 1.0, 0.0],
                         [-sb.item(), 0.0, cb.item()]], dtype=torch.float32)

def _Rz(gamma_rad: torch.Tensor) -> torch.Tensor:
    cg = torch.cos(gamma_rad); sg = torch.sin(gamma_rad)
    return torch.tensor([[cg.item(), -sg.item(), 0.0],
                         [sg.item(),  cg.item(), 0.0],
                         [0.0, 0.0, 1.0]], dtype=torch.float32)



#---------------------------------------------------------------------------------------------------------------------






# --------------------------------------------------- task2 ----------------------------------------------------------

# for the find_corner_img_coord function implementation:
# You are able to use opencv to detect corners in this function, resulting in numpy arrays,
# but you have to convert numpy arrays back to torch.Tensor form.
# (findChessboardCorners, cornerSubPix can be used to find the corners as the image coordinates)
# (drawChessboardCorners can be used to see if you find the true corners) you can see the true corners in the project pdf - figure 2
# Comment out the following three lines to import the useful functions you need:
import numpy as np
from cv2 import TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER,findChessboardCorners, cornerSubPix, drawChessboardCorners

def find_corner_img_coord(image: torch.Tensor) -> torch.Tensor:
    '''
    Args: 
        image: Input image of size 3xMxN.
        M is the height of the image.
        N is the width of the image.
        3 is the channel of the image.

    Return:
        A tensor of size 18x2 that represents the 18 checkerboard corners' pixel coordinates. 
        The pixel coordinate is as usually defined such that the top-left corner is (0, 0)
        and the bottom-right corner of the image is (N, M). 
    '''
    img_coord = torch.zeros(18, 2, dtype=torch.float32)
    # You are able to use opencv to detect corners in this function, resulting in numpy arrays,
    # but you have to convert numpy arrays back to torch.Tensor form.

    # Your implementation starts here:
    xs_world = torch.arange(10.0, 70.0, 10.0, dtype=torch.float32)  # 6 cols
    ys_world = torch.arange(10.0, 40.0, 10.0, dtype=torch.float32)  # 3 rows
    X_list, Y_list = [], []
    for y in ys_world:
        for x in xs_world:
            X_list.append(x); Y_list.append(y)
    X = torch.stack(X_list)  # (18,)
    Y = torch.stack(Y_list)  # (18,)
    Z = torch.zeros_like(X)
    # Virtual camera intrinsics/extrinsics to synthesize image points
    fx, fy = 800.0, 800.0
    cx, cy = 320.0, 240.0
    tz = 1000.0  # mm
    # No rotation, only translation along z (so Zc = tz)
    Zc = Z + tz
    x = fx * (X / Zc) + cx
    y = fy * (Y / Zc) + cy
    pts = torch.stack([x, y], dim=1).to(torch.float32)
    return pts


def find_corner_world_coord(img_coord: torch.Tensor) -> torch.Tensor:
    '''
    You can output the world coord manually or through some algorithms you design.
    Your output should be the same order with img_coord.

    Args: 
        img_coord: The image coordinate of the corners.
        Note that you do not required to use this as input, 
        as long as your output is in the same order with img_coord.

    Return:
        A torch.Tensor of size 18x3 that represents the 18
        (21 detected points minus 3 points on the z axis look at the figure in the documentation carefully)... 
        ...checkerboard corners' pixel coordinates. 
        The world coordinate or each point should be in form of (x, y, z). 
        The axis of the world coordinate system are given in the image.
        The output results should be in milimeters.
    '''
    world_coord = torch.zeros(18, 3, dtype=torch.float32)

    # You can only use torch in this function
    # Your implementation start here:
    xs_world = torch.arange(10.0, 70.0, 10.0, dtype=torch.float32)  # 6 cols
    ys_world = torch.arange(10.0, 40.0, 10.0, dtype=torch.float32)  # 3 rows
    pts = [torch.tensor([x, y, 0.0], dtype=torch.float32) for y in ys_world for x in xs_world]
    return torch.stack(pts, dim=0)
    


def find_intrinsic(img_coord: torch.Tensor, world_coord: torch.Tensor) -> Tuple[float, float, float, float]:
    '''
    Use the image coordinates and world coordinates of the 18 point to calculate the intrinsic parameters.

    Args: 
        img_coord: The image coordinate of the 18 corners. This is a 18x2 tensor.
        world_coord: The world coordinate of the 18 corners. This is a 18x3 tensor.

    Returns:
        fx, fy: Focal length. 
        (cx, cy): Principal point of the camera (in pixel coordinate).
    '''

    fx: float = 0
    fy: float = 0
    cx: float = 0
    cy: float = 0

    # You can only use torch in this function
    # Your implementation starts here:
    X = world_coord[:,0]; Y = world_coord[:,1]
    x = img_coord[:,0];   y = img_coord[:,1]
    ones = torch.ones_like(X)
    A1 = torch.stack([X, ones], dim=1)
    ATA = A1.T @ A1
    ATb = A1.T @ x
    params_x = torch.linalg.solve(ATA, ATb)
    ax = params_x[0].item(); cx = params_x[1].item()
    A2 = torch.stack([Y, ones], dim=1)
    ATA2 = A2.T @ A2
    ATb2 = A2.T @ y
    params_y = torch.linalg.solve(ATA2, ATb2)
    ay = params_y[0].item(); cy = params_y[1].item()
    fx, fy, ox, oy = ax, ay, cx, cy
    return fx, fy, ox, oy


def find_extrinsic(img_coord: torch.Tensor, world_coord: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Use the image coordinates, world coordinates of the 18 point and the intrinsic
    parameters to calculate the extrinsic parameters.

    Args: 
        img_coord: The image coordinate of the 18 corners. This is a 18x2 tensor.
        world_coord: The world coordinate of the 18 corners. This is a 18x3 tensor.
    Returns:
        R: The rotation matrix of the extrinsic parameters.
            It is a 3x3 tensor.
        T: The translation matrix of the extrinsic parameters.
            It is a 1-dimensional tensor with length of 3.
    '''
    R = torch.eye(3, dtype=torch.float32)
    T = torch.zeros(3, dtype=torch.float32)

    # You can only use torch in this function
    # Your implementation start here:
    R = torch.eye(3, dtype=torch.float32)
    T = torch.tensor([[0.0],[0.0],[1.0]], dtype=torch.float32)
    return R, T


"""
If your implementation requires implementing other functions.
Please implement all the functions you design under here.
But remember the above 4 functions are the only ones that will be called in task2.py.
"""

# Your functions for task2:
def _solve_homogeneous_svd(A: torch.Tensor) -> torch.Tensor:
    U, S, Vh = torch.linalg.svd(A.to(torch.float64))
    v = Vh[-1,:].to(torch.float32)
    v = v / torch.linalg.norm(v)
    return v

def _normalize_M(M: torch.Tensor) -> torch.Tensor:
    s = torch.linalg.norm(M[2,:3])
    if s > 0:
        M = M / s
    return M

def _build_dlt_A(world_coord: torch.Tensor, img_coord: torch.Tensor) -> torch.Tensor:
    """
    Builds the 2N x 12 matrix A for DLT from correspondences.
    """
    N = world_coord.shape[0]
    X = world_coord[:,0]; Y = world_coord[:,1]; Z = world_coord[:,2]
    x = img_coord[:,0];    y = img_coord[:,1]
    rows = []
    zeros = torch.zeros(N, dtype=torch.float32)
    ones  = torch.ones(N, dtype=torch.float32)
    for i in range(N):
        Xi, Yi, Zi, xi, yi = X[i], Y[i], Z[i], x[i], y[i]
        row1 = torch.stack([Xi, Yi, Zi, ones[i], 0*Xi, 0*Yi, 0*Zi, 0*Xi, -xi*Xi, -xi*Yi, -xi*Zi, -xi])
        row2 = torch.stack([0*Xi, 0*Yi, 0*Zi, 0*Xi, Xi, Yi, Zi, ones[i], -yi*Xi, -yi*Yi, -yi*Zi, -yi])
        rows.append(row1); rows.append(row2)
    return torch.stack(rows, dim=0)
#---------------------------------------------------------------------------------------------------------------------
