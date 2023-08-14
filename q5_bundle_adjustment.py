import numpy as np
import matplotlib.pyplot as plt

from helper import displayEpipolarF, calc_epi_error, toHomogenous
from q2_1_eightpoint import eightpoint
from q2_2_sevenpoint import sevenpoint
from q3_2_triangulate import findM2

from scipy.optimize import minimize

# Insert your package here


# Helper functions for this assignment. DO NOT MODIFY!!!
"""
Helper functions.

Written by Chen Kong, 2018.
Modified by Zhengyi (Zen) Luo, 2021
"""
def plot_3D_dual(P_before, P_after):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Blue: before; red: after")
    ax.scatter(P_before[:,0], P_before[:,1], P_before[:,2], c = 'blue')
    ax.scatter(P_after[:,0], P_after[:,1], P_after[:,2], c='red')
    while True:
        x, y = plt.ginput(1, mouse_stop=2)[0]
        plt.draw()


'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
            nIters, Number of iterations of the Ransac
            tol, tolerence for inliers
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers

    Hints:
    (1) You can use the calc_epi_error from q1 with threshold to calcualte inliers. Tune the threshold based on 
        the results/expected number of inliners. You can also define your own metric. 
    (2) Use the seven point alogrithm to estimate the fundamental matrix as done in q1
    (3) Choose the resulting F that has the most number of inliers
    (4) You can increase the nIters to bigger/smaller values
 
'''
def ransacF(pts1, pts2, M, nIters=100, tol=10):
    # TODO: Replace pass by your implementation

    inliers_prevcount = 0
    pts1_homo = toHomogenous(pts1)
    pts2_homo = toHomogenous(pts2)

    for i in range(nIters):
        # print("Iteration number:", nIters)
        eight_points = np.random.choice(pts1.shape[0], 8, replace = False)

        pts1_selected = pts1[eight_points]
        pts2_selected = pts2[eight_points]

        F = eightpoint(pts1_selected, pts2_selected, M)

        err = calc_epi_error(pts1_homo, pts2_homo, F)

        inliers = err < tol
        inliers_count = np.sum(inliers)
        if inliers_count > inliers_prevcount:
            inliers_prev = inliers
            inliers_prevcount = inliers_count
        
        
    F_best = eightpoint(pts1[np.where(inliers_prev)], pts2[np.where(inliers_prev)], M)
    return F_best, inliers_prev
    # print(inliers_prevcount)
    # seven_points = np.random.choice(pts1.shape[0], 7, replace = False)

        # pts1_selected = pts1[seven_points]
        # pts2_selected = pts2[seven_points]

        # pts1_homo_selected = toHomogenous(pts1_selected)
        # pts2_homo_selected = toHomogenous(pts2_selected)
        # F = sevenpoint(pts1_selected, pts2_selected, M)

        # for j in range(len(F)):
        #     if isinstance(F, complex):
        #         continue
        #     else:
        #         ress = calc_epi_error(pts1_homo, pts2_homo, F[j])

        #         inliers = ress < tol
        #         inliers_count = np.sum(inliers)
        #         print(inliers_count)

        #         if inliers_count > inliers_prevcount:
        #             F_best = F[j]
        #             inliers_prev = inliers
        #             inliers_prevcount = inliers_count
    


'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # TODO: Replace pass by your implementation

    theta = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)

    if theta == 0:
        R = np.identity(3)
    else:
        u = r/theta
        u_x = np.array([[0, -u[2], u[1]], 
                        [u[2], 0, -u[0]], 
                        [-u[1], u[0], 0]])
        
        u = u.reshape((3,1))
        R = np.eye(3) * np.cos(theta) + (1 - np.cos(theta)) * u @ u.T + u_x * np.sin(theta)

    return R


'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # TODO: Replace pass by your implementation
    A = (R - R.T)/2
    rho = np.array([[A[2,1]], [A[0,2]], [A[1,0]]])
    s = np.sqrt(rho[0]**2 + rho[1]**2 + rho[2]**2)
    c = (R[0,0] + R[1,1] + R[2,2] - 1)/2

    if s == 0 and c == 1:
        r = np.zeros((3,1))

    elif s == 0 and c == -1:
        temp = R + np.identity(3)
        for i in range(temp.shape[1]):
            if np.sum(temp[:,i]) != 0:
                v = temp[:, i]
                break

        u = v/np.linalg.norm(v)
        r_tmp = u * np.pi()
        if (np.linalg.norm(r_tmp) == np.pi()) and ((r_tmp[0,:] == 0 and r_tmp[1,:] == 0 
            and r_tmp[2,:] < 0) or (r_tmp[0,:] == 0 and r_tmp[1,:] < 0) or (r_tmp[0,:] < 0)):  
            r = -r_tmp

        else:
            r = r_tmp
    else:
        u = rho/s
        theta = np.arctan2(s, c)
        r = u * theta 

    return r.flatten()


'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenation of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # TODO: Replace pass by your implementation
    N = p1.shape[0]
    P, r2, t2 = x[0:-6], x[-6:-3], x[-3:]
    P = np.reshape(P, (N,3))
    P_homo = np.vstack((P.T, np.ones((1,N)))).T
    
    R = rodrigues(r2)
    M2 = np.column_stack((R, t2))

    C1 = K1.dot(M1)
    C2 = K2.dot(M2)

    p1_hat = (C1 @ P_homo.T).T
    p1_hatz = p1_hat[:,2].reshape((p1_hat.shape[0],1))
    p1_hat = p1_hat/p1_hatz

    p2_hat = (C2 @ P_homo.T).T
    p2_hatz = p2_hat[:,2].reshape((p2_hat.shape[0],1))
    p2_hat = p2_hat/p2_hatz

    residuals = np.concatenate([(p1-p1_hat[:,:2]).reshape([-1]),(p2-p2_hat[:,:2]).reshape([-1])])

    return residuals

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
            o1, the starting objective function value with the initial input
            o2, the ending objective function value after bundle adjustment

    Hints:
    (1) Use the scipy.optimize.minimize function to minimize the objective function, rodriguesResidual. 
        You can try different (method='..') in scipy.optimize.minimize for best results. 
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    
    obj_start = obj_end = 0
    # ----- TODO -----
    # YOUR CODE HERE
    R = M2_init[:,:3]
    t = M2_init[:,3]

    r = invRodrigues(R)
    x = np.hstack((P_init.flatten(), r.flatten(), t.flatten()))

    objective = lambda x: (rodriguesResidual(K1, M1, p1, K2, p2, x)**2).sum()
    obj_start = objective(x)
    residuals = minimize(objective, x)

    P, r2, t2 = residuals.x[0:-6], residuals.x[-6:-3], residuals.x[-3:]
    P = P.reshape((p1.shape[0],3))

    R_opt = rodrigues(r2)
    M2 = np.column_stack((R_opt, t2))

    obj_end = residuals.fun

    return M2, P, obj_start, obj_end



if __name__ == "__main__":
              
    np.random.seed(1) #Added for testing, can be commented out

    some_corresp_noisy = np.load('data/some_corresp_noisy.npz') # Loading correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinscis of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    noisy_pts1, noisy_pts2 = some_corresp_noisy['pts1'], some_corresp_noisy['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')

    F_eightpoint = eightpoint(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]))
    F, inliers = ransacF(noisy_pts1, noisy_pts2, M=np.max([*im1.shape, *im2.shape]))

    # displayEpipolarF(im1, im2, F_eightpoint)

    # Simple Tests to verify your implementation:
    pts1_homogenous, pts2_homogenous = toHomogenous(noisy_pts1), toHomogenous(noisy_pts2)

    assert(F.shape == (3, 3))
    assert(F[2, 2] == 1)
    assert(np.linalg.matrix_rank(F) == 2)
    
    # Simple Tests to verify your implementation:
    from scipy.spatial.transform import Rotation as sRot
    rotVec = sRot.random()
    mat = rodrigues(rotVec.as_rotvec())
    
    # print(np.linalg.det(mat))

    assert(np.linalg.norm(rotVec.as_rotvec() - invRodrigues(mat)) < 1e-3)
    assert(np.linalg.norm(rotVec.as_matrix() - mat) < 1e-3)

    # Visualization:
    np.random.seed(1)
    correspondence = np.load('data/some_corresp_noisy.npz') # Loading noisy correspondences
    intrinsics = np.load('data/intrinsics.npz') # Loading the intrinsics of the camera
    K1, K2 = intrinsics['K1'], intrinsics['K2']
    pts1, pts2 = correspondence['pts1'], correspondence['pts2']
    im1 = plt.imread('data/im1.png')
    im2 = plt.imread('data/im2.png')
    M=np.max([*im1.shape, *im2.shape])

    # TODO: YOUR CODE HERE
    '''
    Call the ransacF function to find the fundamental matrix
    Call the findM2 function to find the extrinsics of the second camera
    Call the bundleAdjustment function to optimize the extrinsics and 3D points
    Plot the 3D points before and after bundle adjustment using the plot_3D_dual function
    '''
    M1 = np.column_stack((np.eye(3), np.zeros((3,1))))
    F, inliers = ransacF(pts1, pts2, M)
    M2_init, C2, P_init = findM2(F, pts1[np.where(inliers)], pts2[np.where(inliers)], intrinsics, filename = 'q5_3.npz')
    M2, P_final, o1, o2 = bundleAdjustment(K1, M1, pts1[np.where(inliers)], K2, M2_init, pts2[np.where(inliers)], P_init)

    print("Initial Residual Error", o1)
    print("Final Residual Error:", o2)
    plot_3D_dual(P_init, P_final)
