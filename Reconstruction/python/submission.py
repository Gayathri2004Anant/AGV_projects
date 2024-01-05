"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
import cv2 as cv
import numpy.linalg
import helper as hp


def homogenise(arr):
    if len(arr.shape) != 2:
        raise ValueError("Input array must be 2-dimensional.")

    # Create an array of ones with the same number of rows as the input array
    ones_array = np.ones((arr.shape[0], 1))

    # Stack the ones_array as a new column to the input array
    result = np.column_stack((arr, ones_array))

    return result


"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):
    # Left image points
    x1, y1 = pts1[:, 0], pts1[:, 1]

    # Right image points
    x2, y2 = pts2[:, 0], pts2[:, 1]

    # Normalizing matrix
    T = np.array([[1 / M, 0, 0],
                  [0, 1 / M, 0],
                  [0, 0, 1]])

    # Normalizing coordinates in left and right image
    M=float(M)
    x1 = x1/ M
    x2 = x2/ M
    y1 = y1/M
    y2 = y2/ M

    # Coefficient matrix
    one = np.ones_like(x1)
    U = np.column_stack((x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, one))

    # F is the eigenvector corresponding to the smallest eigenvalue of U
    _, _, V = np.linalg.svd(U)

    F = V[-1].reshape(3, 3)

    # Rank 2 constraint
    A, B, C = np.linalg.svd(F)
    B[2] = 0
    F = A @ np.diag(B) @ C

    # Denormalizing
    F = np.dot(np.dot(T.T, F), T)

    # Refining F
    F = hp.refineF(F, pts1, pts2)

    return F

def calculate_ssd(img1, img2):
    """Computing the sum of squared differences (SSD) between two images."""
    if img1.shape != img2.shape:
        print("Images don't have the same shape.",img1.shape,img2.shape)
        return
    return np.sum((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32))**2)


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""

def epipolar_correspondences(im1, im2, F, pts1):
    pts_x=pts1[:,0].T
    pts_y=pts1[:,1].T
    pts1=homogenise(pts1).T
    print(np.shape(F), np.shape(pts1))
    L= F @ pts1
    print(np.shape(L))
    pts2=[]
    h, w, _=np.shape(im1)
    im1=cv.cvtColor(im1,cv.COLOR_BGR2GRAY)
    im2=cv.cvtColor(im2,cv.COLOR_BGR2GRAY)
    r, c=np.shape(L)
    window=10

    for i in range(c):
        min_x =0
        min_y=0
        min = float('inf')
        l=L[:,i]
        np.reshape(l, (1, 3))
        x = window
        while x + window < w:
            y = int((-l[2] - l[0] * x) / l[1])
            if window<=y<=h-window:
                block1=im1[pts_y[i]-window:pts_y[i]+window+1, pts_x[i]-window:pts_x[i]+window+1]
                block2=im2[y-window:y+window+1, x-window:x+window+1]
                ssd=calculate_ssd(block1,block2)
                print(x,ssd)
                if ssd<min:
                    min=ssd
                    min_x=x
                    min_y=y
                # print(min)
            x+=1
            print(min_x,min_y,min)
        pts2.append([min_x,min_y])
    pts2=np.array(pts2)
    return pts2

"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    E=(K2.T)@F@K1
    return E


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    # replace pass by your implementation
    num_cor=pts1.shape[0]
    P = np.zeros((num_cor, 3))
    err = 0
    for i in range(num_cor):
        x=pts1[i][0]
        y=pts1[i][1]
        x_=pts2[i][0]
        y_=pts2[i][1]

        A = [y * P1[2, :] - P1[1, :],
             P1[0, :] - x * P1[2, :],
             y_ * P2[2, :] - P2[1, :],
             P2[0, :] - x_ * P2[2, :]
             ]

        U, s, V = np.linalg.svd(A, full_matrices=False)

        P_unnormalized = V[-1, :]
        P_normalized = P_unnormalized / P_unnormalized[3]
        P[i, :] = P_normalized[:3]

        # Reprojecting to 2D image
        p1_pred = np.dot(P1, P_normalized)
        p1_pred = p1_pred / p1_pred[2]
        p2_pred = np.dot(P2, P_normalized)
        p2_pred = p2_pred / p2_pred[2]

        # Reprojection error
        err += np.linalg.norm(pts1[i, :2] - p1_pred[:2])+np.linalg.norm(pts2[i, :2] - p2_pred[:2])
    err/=(2*num_cor)
    return P, err, not np.any(P[:,2]<0)

"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    pass


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    # replace pass by your implementation
    pass


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    pass


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation
    pass


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # replace pass by your implementation
    pass
