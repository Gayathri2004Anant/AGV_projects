import cv2 as cv
import numpy as np
import helper as hp
import submission as sub
import matplotlib.pyplot as plt


data=np.load('../data/some_corresp.npz')
pts1=data['pts1']
pts2=data['pts2']
pts1_h=sub.homogenise(pts1)
pts2_h=sub.homogenise(pts2)

dataTemple=np.load('../data/temple_coords.npz')
pts1Temple=data['pts1']
pts2Temple=data['pts2']


intrinsics=np.load('../data/intrinsics.npz')
K1=intrinsics['K1']
K2=intrinsics['K2']


img1=cv.imread('../data/im1.png')
img2=cv.imread('../data/im2.png')
h,w,_=np.shape(img1)
h2,w2,__=np.shape(img2)
print(h,w)
print(h2,w2)
M=max(h,w)

F=sub.eight_point(pts1,pts2,M)

hp.displayEpipolarF(img1,img2,F)
hp.epipolarMatchGUI(img1,img2,F)

E=sub.essential_matrix(F,K1,K2)
# print(E)

M1=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])

M2s=hp.camera2(E)
print('------------------------------------------')

for i in range(4):
    M2=M2s[:,:,i]
    P1=K1@M1
    P2=K2@M2
    pts,err,positive=sub.triangulate(P1,pts1Temple,P2,pts2Temple)
    print(err,positive)
    x=pts[:,0]
    y=pts[:,1]
    z=pts[:,2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='b', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()



