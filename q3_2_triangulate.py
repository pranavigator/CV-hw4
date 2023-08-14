import numpy as np
import matplotlib.pyplot as plt

from helper import camera2
from q2_1_eightpoint import eightpoint
from q3_1_essential_matrix import essentialMatrix

# Insert your package here


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: w, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.

    Hints:
    (1) For every input point, form A using the corresponding points from pts1 & pts2 and C1 & C2
    (2) Solve for the least square solution using np.linalg.svd
    (3) Calculate the reprojection error using the calculated 3D points and C1 & C2 (do not forget to convert from 
        homogeneous coordinates to non-homogeneous ones)
    (4) Keep track of the 3D points and projection error, and continue to next point 
    (5) You do not need to follow the exact procedure above. 
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    # ----- TODO -----
    # YOUR CODE HERE

    err = 0
    w = np.empty((pts1.shape[0], 3))

    for i in range(pts1.shape[0]):
        A_i = np.array([pts1[i,1] * C1[2,:].T - C1[1,:].T, 
                        C1[0,:].T - pts1[i,0]*C1[2,:].T, 
                        pts2[i,1] * C2[2,:].T - C2[1,:].T, 
                        C2[0,:].T - pts2[i,0]*C2[2,:].T])
        
        _, _, VT = np.linalg.svd(A_i)

        wi_tilda = VT[-1,:]

        x1i_hat = C1 @ wi_tilda
        x1i_hat = x1i_hat / x1i_hat[2]

        x2i_hat = C2 @ wi_tilda
        x2i_hat = x2i_hat / x2i_hat[2]

        err = err + np.linalg.norm(pts1[i] - x1i_hat[:2])**2 + np.linalg.norm(pts2[i] - x2i_hat[:2])**2

        wi_tilda = wi_tilda/wi_tilda[3]
        
        w[i, :] = wi_tilda[:3]
    
    return w, err


'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

'''
Q3.2: Function to find camera2's projective matrix given correspondences
    Input:  F, the pre-computed fundamental matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            pts2, the Nx2 matrix with the 2D image coordinates per row
            intrinsics, the intrinsics of the cameras, load from the .npz file
            filename, the filename to store results
    Output: [M2, C2, w] the computed M2 (3x4) camera projective matrix, C2 (3x4) K2 * M2, and the 3D points P (Nx3)

***
Hints:
(1) Loop through the 'M2s' and use triangulate to calculate the 3D points and projection error. Keep track 
    of the projection error through best_error and retain the best one. 
(2) Remember to take a look at camera2 to see how to correctly reterive the M2 matrix from 'M2s'. 

'''
def findM2(F, pts1, pts2, intrinsics, filename = 'q3_3.npz'):
    # ----- TODO -----
    # YOUR CODE HERE
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    
    E = essentialMatrix(F, K1, K2)

    M2s = camera2(E)

    M1 = np.hstack((np.identity(3), np.zeros(3)[:,np.newaxis]))
    C1 = K1.dot(M1)

    # err_prev = 10000

    for i in range(M2s.shape[2]):
        M2_iter = M2s[:,:,i]
        C2_iter = K2.dot(M2_iter)
        w_iter, err = triangulate(C1, pts1, C2_iter, pts2)
        if np.all(w_iter[:,2] > 0):
            M2 = M2_iter
            C2 = C2_iter
            w = w_iter
    
    np.savez(filename, M2, C2, w)
    return M2, C2, w



if __name__ == "__main__":

    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    M2, C2, w = findM2(F, pts1, pts2, intrinsics)
    # np.savez('q3_3.npz', M2, C2, w)
    
    # Simple Tests to verify your implementation:
    M1 = np.hstack((np.identity(3), np.zeros(3)[:,np.newaxis]))
    C1 = K1.dot(M1)
    C2 = K2.dot(M2)
    P_test, err = triangulate(C1, pts1, C2, pts2)
    print(err)
    assert(err < 500)