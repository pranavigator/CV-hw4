import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous, _singularize, refineF

# Insert your package here


'''
Q2.2: Seven Point Algorithm for calculating the fundamental matrix
    Input:  pts1, 7x2 Matrix containing the corresponding points from image1
            pts2, 7x2 Matrix containing the corresponding points from image2
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated 3x3 fundamental matrixes.
    
    HINTS:
    (1) Normalize the input pts1 and pts2 scale paramter M.
    (2) Setup the seven point algorithm's equation.
    (3) Solve for the least square solution using SVD. 
    (4) Pick the last two column vector of vT.T (the two null space solution f1 and f2)
    (5) Use the singularity constraint to solve for the cubic polynomial equation of  F = a*f1 + (1-a)*f2 that leads to 
        det(F) = 0. Solving this polynomial will give you one or three real solutions of the fundamental matrix. 
        Use np.polynomial.polynomial.polyroots to solve for the roots
    (6) Unscale the fundamental matrixes and return as Farray
'''
def sevenpoint(pts1, pts2, M):

    Farray = []
    # ----- TODO -----
    # YOUR CODE HERE
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

    A[:, 0] = pts2_scaled[:, 0] *    pts1_scaled[:,0]
    A[:, 1] = pts2_scaled[:, 0] * pts1_scaled[:,1]
    A[:, 2] = pts2_scaled[:, 0]
    A[:, 3] = pts2_scaled[:, 1] * pts1_scaled[:,0]
    A[:, 4] = pts2_scaled[:, 1] * pts1_scaled[:,1]
    A[:, 5] = pts2_scaled[:, 1]
    A[:, 6] = pts1_scaled[:,0]
    A[:, 7] = pts1_scaled[:,1]

    U, S, VT = np.linalg.svd(A)
    F1 = VT[-1, :]
    F2 = VT[-2, :]

    F1 = np.reshape(F1, (3,3))
    F2 = np.reshape(F2, (3,3))

    #Setting 4 random values for a
    a = np.array([0, 0.25, 0.5, 0.75])
    f1 = np.linalg.det(a[0] * F1 + (1 - a[0]) * F2)
    f2 = np.linalg.det(a[1] * F1 + (1 - a[1]) * F2)
    f3 = np.linalg.det(a[2] * F1 + (1 - a[2]) * F2)
    f4 = np.linalg.det(a[3] * F1 + (1 - a[3]) * F2)

    f = np.array([f1, f2, f3, f4]).T
    a_poly = np.array([[1, a[0], a[0]**2, a[0]**3], 
                       [1, a[1], a[1]**2, a[1]**3],
                       [1, a[2], a[2]**2, a[2]**3],
                       [1, a[3], a[3]**2, a[3]**3]])
    
    coeffs = np.linalg.inv(a_poly) @ f
    # print("coeffs:", coeffs)

    a = np.polynomial.polynomial.polyroots(coeffs)
    # print("a:", a.shape)
    
    for i in range(len(a)):
        #Ensuring that only the real parts of the roots are being considered
        #by using np.real()
        F = float(np.real(a[i])) * F1 + (1 - float(np.real(a[i]))) * F2
        F = refineF(F, pts1_scaled, pts2_scaled)
        F = T.T @ F @ T
        F = F/F[2,2]
        Farray.append(F)
    
    return Farray



if __name__ == "__main__":
        
    correspondence = np.load('data/some_corresp.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    # indices = np.arange(pts1.shape[0])
    # indices = np.random.choice(indices, 7, False)
    indices = np.array([82, 19, 56, 84, 54, 24, 18])

    M = np.max([*im1.shape, *im2.shape])

    Farray = sevenpoint(pts1[indices, :], pts2[indices, :], M)

    F = Farray[0]

    np.savez('q2_2.npz', F, M)

    # Simple Tests to verify your implementation:
    # Test out the seven-point algorithm by randomly sampling 7 points and finding the best solution. 
    np.random.seed(1) #Added for testing, can be commented out
    
    pts1_homogenous, pts2_homogenous = toHomogenous(pts1), toHomogenous(pts2)

    max_iter = 500
    pts1_homo = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    pts2_homo = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    ress = []
    F_res = []
    choices = []
    M=np.max([*im1.shape, *im2.shape])
    for i in range(max_iter):
        choice = np.random.choice(range(pts1.shape[0]), 7)
        pts1_choice = pts1[choice, :]
        pts2_choice = pts2[choice, :]
        Fs = sevenpoint(pts1_choice, pts2_choice, M)
        for F in Fs:
            choices.append(choice)
            res = calc_epi_error(pts1_homo,pts2_homo, F)
            F_res.append(F)
            ress.append(np.mean(res))
            
    min_idx = np.argmin(np.abs(np.array(ress)))
    F = F_res[min_idx]
    print("F:", F)
    print("Error:", ress[min_idx])

    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    assert(np.mean(calc_epi_error(pts1_homogenous, pts2_homogenous, F)) < 1)

    # fundamental matrix must have rank 2!
    # assert(np.linalg.matrix_rank(F) == 2)
    displayEpipolarF(im1, im2, F)