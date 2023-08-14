import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, refineF, _singularize

# Insert your package here



'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix

    HINTS:
    (1) Normalize the input pts1 and pts2 using the matrix T.
    (2) Setup the eight point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Use the function `_singularize` (provided) to enforce the singularity condition. 
    (5) Use the function `refineF` (provided) to refine the computed fundamental matrix. 
        (Remember to use the normalized points instead of the original points)
    (6) Unscale the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    # ----- TODO -----
    # YOUR CODE HERE
    # pts1_scaled = np.zeros(pts1.shape)
    # pts2_scaled = np.zeros(pts2.shape)

    pts1_homogenous = toHomogenous(pts1)
    pts2_homogenous = toHomogenous(pts2)

    T = np.array([[1/M,  0,   0],
                  [0   , 1/M, 0],
                  [0  ,  0 ,   1]])
    
    pts1_scaled_homo = pts1_homogenous @ T
    pts1_scaled = pts1_scaled_homo[:,:2]

    pts2_scaled_homo = pts2_homogenous @ T
    pts2_scaled = pts2_scaled_homo[:,:2]
    
    A = np.ones((pts1.shape[0], 9))

    A[:, 0] = pts2_scaled[:, 0] * pts1_scaled[:,0]
    A[:, 1] = pts2_scaled[:, 0] * pts1_scaled[:,1]
    A[:, 2] = pts2_scaled[:, 0]
    A[:, 3] = pts2_scaled[:, 1] * pts1_scaled[:,0]
    A[:, 4] = pts2_scaled[:, 1] * pts1_scaled[:,1]
    A[:, 5] = pts2_scaled[:, 1]
    A[:, 6] = pts1_scaled[:,0]
    A[:, 7] = pts1_scaled[:,1]

    U, S, VT = np.linalg.svd(A)
    F = VT[-1, :]
    # print()
    F = np.reshape(F, (3,3))
    
    F = _singularize(F)
    F = refineF(F, pts1_scaled, pts2_scaled)

    #Unscaling F
    F = T.T @ F @ T
    F = F/F[2,2]

    # print(F)
    return F




if __name__ == "__main__":
        
    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    F = eightpoint(pts1, pts2, M=np.max([*im1.shape, *im2.shape]))

    np.savez('q2_1.npz', F, M=np.max([*im1.shape, *im2.shape]))

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    print(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)))
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)

    # Q2.1
    displayEpipolarF(im1, im2, F)