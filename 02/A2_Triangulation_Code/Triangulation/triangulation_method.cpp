/**
 * Copyright (C) 2015 by Liangliang Nan (liangliang.nan@gmail.com)
 * https://3d.bk.tudelft.nl/liangliang/
 *
 * This file is part of Easy3D. If it is useful in your research/work,
 * I would be grateful if you show your appreciation by citing it:
 * ------------------------------------------------------------------
 *      Liangliang Nan.
 *      Easy3D: a lightweight, easy-to-use, and efficient C++
 *      library for processing and rendering 3D data. 2018.
 * ------------------------------------------------------------------
 * Easy3D is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License Version 3
 * as published by the Free Software Foundation.
 *
 * Easy3D is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include "triangulation.h"
#include "matrix_algo.h"
#include <easy3d/optimizer/optimizer_lm.h>
#include <numeric>
#include <cmath>

using namespace easy3d;

static bool solve_linear_system(const Matrix& A, const Vector& b, Vector& x);

/**
 * Normalize a set of 2D points for numerical stability.
 * @param points Original 2D points
 * @param norm_points Output normalized points
 * @param T Output transformation matrix
 */
void norm_transformation(const std::vector<Vector2D>& points, std::vector<Vector2D>& norm_points, Matrix33& T) {
    Vector2D centroid(0, 0);
    int n = points.size();
    for (const auto& p: points) {
        centroid += p;
    }
    centroid /= n;

    double dis = 0.0;
    for (const auto& p: points) {
        dis += (centroid - p).length();
    }
    double scale = sqrt(2) / (dis/n);
    Vector2D norm_centroid = scale*centroid;

    T = Matrix33(
        scale, 0, -scale * centroid.x(),
        0, scale, -scale * centroid.y(),
        0, 0, 1
    );

    norm_points.resize(n);
    for (int i = 0; i < n; i++) {
        norm_points[i] = (scale * points[i]) - norm_centroid;
    }
}

/**
 * Calculate matrix W for fundamental matrix estimation
 * @param n Number of point correspondences
 * @param points_0 Points from first image
 * @param points_1 Points from second image
 * @return Matrix W used in the 8-point algorithm
 */
Matrix cal_matrix_W(int n, const std::vector<Vector2D>& points_0, const std::vector<Vector2D>& points_1) {
    Matrix W(n, 9);
    for (int i = 0; i < n; i++) {
        double u0 = points_0[i].x(), v0 = points_0[i].y(), u1 = points_1[i].x(), v1 = points_1[i].y();
        W.set_row(i, {
            u1*u0, u1*v0, u1,
            v1*u0, v1*v0, v1,
            u0, v0, 1
        });
    }
    return W;
}

/**
 * Estimate the fundamental matrix using the normalized 8-point algorithm
 * @param points_0 Points from first image
 * @param points_1 Points from second image
 * @return Estimated fundamental matrix
 */
Matrix33 estimate_fundamental_matrix(const std::vector<Vector2D>& points_0, const std::vector<Vector2D>& points_1) {
    // Normalize points
    std::vector<Vector2D> norm_points_0, norm_points_1;
    Matrix33 T0, T1;
    norm_transformation(points_0, norm_points_0, T0);
    norm_transformation(points_1, norm_points_1, T1);

    // Compute W matrix
    int n = points_0.size();
    Matrix W = cal_matrix_W(n, norm_points_0, norm_points_1);

    // Apply SVD to find F
    Matrix U(n, n, 0.0), S(n, 9, 0.0), V(9, 9, 0.0);
    svd_decompose(W, U, S, V);

    // Extract F from the last column of V
    Matrix F(3, 3, V.get_column(8).data());

    // Enforce rank-2 constraint
    Matrix U_f(3, 3, 0.0), V_f(3, 3, 0.0);
    Matrix D_f(3, 3, 0.0);
    svd_decompose(F, U_f, D_f, V_f);
    D_f.set(2, 2, 0.0);  // Force smallest singular value to zero

    // Reconstruct F with rank 2
    F = U_f * D_f * V_f;

    // Denormalize
    F = T1.transpose() * F * T0;

    return F;
}

/**
 * Extract essential matrix from fundamental matrix
 * @param F Fundamental matrix
 * @param K Camera intrinsic matrix
 * @return Essential matrix
 */
Matrix33 compute_essential_matrix(const Matrix33& F, const Matrix33& K) {
    return K.transpose() * F * K;
}

/**
 * Extract possible rotation and translation from essential matrix
 * @param E Essential matrix
 * @param rotations Output vector of possible rotation matrices
 * @param translations Output vector of possible translation vectors
 */
void extract_camera_poses(const Matrix33& E, std::vector<Matrix33>& rotations, std::vector<Vector3D>& translations) {
    // Perform SVD on E
    Matrix U(3, 3, 0.0), V(3, 3, 0.0);
    Matrix S(3, 3, 0.0);
    svd_decompose(E, U, S, V);

    // Define the W matrix
    Matrix33 W({
        0, -1, 0,
        1, 0, 0,
        0, 0, 1
    });

    // Four possible solutions for R and t
    Matrix33 R1 = U * W * V.transpose();
    Matrix33 R2 = U * W.transpose() * V.transpose();

    // Ensure R has positive determinant (rotation matrix property)
    if (determinant(R1) < 0) R1 = -R1;
    if (determinant(R2) < 0) R2 = -R2;

    // Translation is the last column of U (up to scale)
    Vector3D t1(U.get_column(2)[0], U.get_column(2)[1], U.get_column(2)[2]);
    Vector3D t2 = -t1;

    // Return all combinations
    rotations = {R1, R1, R2, R2};
    translations = {t1, t2, t1, t2};
}

/**
 * Determine the correct camera pose by checking point depths
 * @param rotations Possible rotation matrices
 * @param translations Possible translation vectors
 * @param K Camera intrinsic matrix
 * @param points_0 Points from first image
 * @param points_1 Points from second image
 * @param best_R Output best rotation matrix
 * @param best_t Output best translation vector
 * @return Index of the best pose
 */
int find_best_pose(
    const std::vector<Matrix33>& rotations,
    const std::vector<Vector3D>& translations,
    const Matrix33& K,
    const std::vector<Vector2D>& points_0,
    const std::vector<Vector2D>& points_1,
    Matrix33& best_R,
    Vector3D& best_t
) {
    int n = points_0.size();
    std::vector<int> counter(4, 0);

    // Check all four possible combinations
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < n; j++) {
            Vector3D temp_P = inverse(K) * points_0[j].homogeneous();
            Vector3D Q = rotations[i] * temp_P + translations[i];
            Vector3D temp_Q = inverse(K) * points_1[j].homogeneous();
            Vector3D P = rotations[i].transpose() * (temp_Q - translations[i]);

            // Count points with positive depth in both cameras
            if (P.z() > 0 && Q.z() > 0) {
                counter[i]++;
            }
        }
    }

    // Find the pose with maximum points in front of both cameras
    auto maxIt = std::max_element(counter.begin(), counter.end());
    int maxIndex = std::distance(counter.begin(), maxIt);

    // Set the best pose
    best_R = rotations[maxIndex];
    best_t = translations[maxIndex];

    std::cout << "R1t1 count= " << counter[0] << std::endl;
    std::cout << "R1t2 count= " << counter[1] << std::endl;
    std::cout << "R2t1 count= " << counter[2] << std::endl;
    std::cout << "R2t2 count= " << counter[3] << std::endl;
    std::cout << "Selected R" << maxIndex/2 + 1 << "t" << maxIndex%2 + 1 << std::endl;

    return maxIndex;
}

/**
 * Linear triangulation of points
 * @param K Camera intrinsic matrix
 * @param R Camera rotation matrix
 * @param t Camera translation vector
 * @param points_0 Points from first image
 * @param points_1 Points from second image
 * @param points_3d Output triangulated 3D points
 * @return True if triangulation was successful
 */
bool linear_triangulation(
    const Matrix33& K,
    const Matrix33& R,
    const Vector3D& t,
    const std::vector<Vector2D>& points_0,
    const std::vector<Vector2D>& points_1,
    std::vector<Vector3D>& points_3d
) {
    int n = points_0.size();
    points_3d.clear();
    points_3d.reserve(n);

    // Create projection matrices
    Matrix33 KR = K * R;
    Matrix M0(3, 4, {
        K[0][0], K[0][1], K[0][2], 0,
        K[1][0], K[1][1], K[1][2], 0,
        K[2][0], K[2][1], K[2][2], 0
    });
    Matrix M1(3, 4, {
        KR[0][0], KR[0][1], KR[0][2], t[0],
        KR[1][0], KR[1][1], KR[1][2], t[1],
        KR[2][0], KR[2][1], KR[2][2], t[2]
    });

    // Triangulate each point pair
    Matrix A(4, 4);
    Matrix U(4, 4, 0.0), S(4, 4, 0.0), V(4, 4, 0.0);

    for (int i = 0; i < n; i++) {
        A.set_row(0, points_0[i].x() * M0.get_row(2) - M0.get_row(0));
        A.set_row(1, points_0[i].y() * M0.get_row(2) - M0.get_row(1));
        A.set_row(2, points_1[i].x() * M1.get_row(2) - M1.get_row(0));
        A.set_row(3, points_1[i].y() * M1.get_row(2) - M1.get_row(1));

        svd_decompose(A, U, S, V);
        Vector4D P_homo = V.get_column(3);
        points_3d.push_back(P_homo.cartesian());
    }

    return true;
}

/**
 * Non-linear refinement of triangulated points using Gauss-Newton method
 * @param M0 Projection matrix for the first camera
 * @param M1 Projection matrix for the second camera
 * @param points_0 Points from first image
 * @param points_1 Points from second image
 * @param points_3d Input/output 3D points to be refined
 * @return True if refinement was successful
 */
bool nonlinear_refinement(
    const Matrix& M0,
    const Matrix& M1,
    const std::vector<Vector2D>& points_0,
    const std::vector<Vector2D>& points_1,
    std::vector<Vector3D>& points_3d
) {
    const int max_iterations = 10;  // Maximum number of iterations
    const double epsilon = 1e-8;    // Convergence threshold

    int num_points = points_3d.size();
    bool success = true;

    // Refine each point individually
    for (int i = 0; i < num_points; ++i) {
        Vector3D& P = points_3d[i];
        Vector2D p0 = points_0[i];
        Vector2D p1 = points_1[i];

        bool converged = false;
        for (int iter = 0; iter < max_iterations && !converged; ++iter) {
            // Project 3D point to both images
            Vector4D P_homo(P.x(), P.y(), P.z(), 1.0);

            // Compute projection in first camera
            Vector3D proj0 = M0 * P_homo;
            Vector2D p0_proj(proj0.x() / proj0.z(), proj0.y() / proj0.z());

            // Compute projection in second camera
            Vector3D proj1 = M1 * P_homo;
            Vector2D p1_proj(proj1.x() / proj1.z(), proj1.y() / proj1.z());

            // Compute residuals
            Vector2D e0 = p0 - p0_proj;
            Vector2D e1 = p1 - p1_proj;

            // Build the Jacobian matrix (2n×3 for n cameras, here n=2)
            Matrix J(4, 3, 0.0);

            // Compute Jacobian for first camera
            double x = proj0.x(), y = proj0.y(), z = proj0.z();
            double z2 = z * z;

            // d(x/z)/dX, d(x/z)/dY, d(x/z)/dZ
            J.set_row(0, {
                M0(0, 0) / z - x * M0(2, 0) / z2,
                M0(0, 1) / z - x * M0(2, 1) / z2,
                M0(0, 2) / z - x * M0(2, 2) / z2
            });

            // d(y/z)/dX, d(y/z)/dY, d(y/z)/dZ
            J.set_row(1, {
                M0(1, 0) / z - y * M0(2, 0) / z2,
                M0(1, 1) / z - y * M0(2, 1) / z2,
                M0(1, 2) / z - y * M0(2, 2) / z2
            });

            // Compute Jacobian for second camera
            x = proj1.x();
            y = proj1.y();
            z = proj1.z();
            z2 = z * z;

            // d(x/z)/dX, d(x/z)/dY, d(x/z)/dZ
            J.set_row(2, {
                M1(0, 0) / z - x * M1(2, 0) / z2,
                M1(0, 1) / z - x * M1(2, 1) / z2,
                M1(0, 2) / z - x * M1(2, 2) / z2
            });

            // d(y/z)/dX, d(y/z)/dY, d(y/z)/dZ
            J.set_row(3, {
                M1(1, 0) / z - y * M1(2, 0) / z2,
                M1(1, 1) / z - y * M1(2, 1) / z2,
                M1(1, 2) / z - y * M1(2, 2) / z2
            });

            // Create residual vector e
            Vector e(4);
            e[0] = e0.x();
            e[1] = e0.y();
            e[2] = e1.x();
            e[3] = e1.y();

            // Compute J^T * J
            Matrix JtJ = J.transpose() * J;

            // Compute J^T * e
            Vector Jte = J.transpose() * e;

            // Solve for the update: (J^T * J) * delta_P = J^T * e
            Vector delta_P(3);
            bool solved = solve_linear_system(JtJ, Jte, delta_P);

            if (!solved) {
                // If the system couldn't be solved, skip this point
                success = false;
                break;
            }

            // Update the 3D point
            P.x() += delta_P[0];
            P.y() += delta_P[1];
            P.z() += delta_P[2];

            // Check for convergence
            if (delta_P.length() < epsilon) {
                converged = true;
            }
        }
    }

    return success;
}


/**
 * Calculate reprojection error for triangulated points
 * @param K Camera intrinsic matrix
 * @param R Camera rotation matrix
 * @param t Camera translation vector
 * @param points_0 Original points from first image
 * @param points_1 Original points from second image
 * @param points_3d Triangulated 3D points
 * @return Average reprojection error
 */
double compute_reprojection_error(
    const Matrix33& K,
    const Matrix33& R,
    const Vector3D& t,
    const std::vector<Vector2D>& points_0,
    const std::vector<Vector2D>& points_1,
    const std::vector<Vector3D>& points_3d
) {
    int n = points_3d.size();
    double total_error = 0.0;

    // Create projection matrices
    Matrix34 P0, P1;
    P0 = Matrix34({
        K(0,0), K(0,1), K(0,2), 0,
        K(1,0), K(1,1), K(1,2), 0,
        K(2,0), K(2,1), K(2,2), 0
    });

    // P1 = K * [R|t]
    Matrix34 Rt(
    R(0,0), R(0,1), R(0,2), t.x(),
    R(1,0), R(1,1), R(1,2), t.y(),
    R(2,0), R(2,1), R(2,2), t.z()
);
    P1 = K * Rt;

    for (int i = 0; i < n; ++i) {
        // Create homogeneous point
        Vector4D P_homo(points_3d[i].x(), points_3d[i].y(), points_3d[i].z(), 1.0);

        // Project to first image
        Vector3D proj0 = P0 * P_homo;
        Vector2D p0_proj(proj0.x() / proj0.z(), proj0.y() / proj0.z());

        // Project to second image
        Vector3D proj1 = P1 * P_homo;
        Vector2D p1_proj(proj1.x() / proj1.z(), proj1.y() / proj1.z());

        // Calculate reprojection error
        double error0 = (points_0[i] - p0_proj).length();
        double error1 = (points_1[i] - p1_proj).length();

        total_error += (error0 + error1);
    }

    return total_error / (2.0 * n);  // Average error per point per image
}

bool Triangulation::triangulation(
        double fx, double fy,     // input: focal lengths (same for both cameras)
        double cx, double cy,     // input: principal point (same for both cameras)
        double s,                 // input: skew factor (same for both cameras)
        const std::vector<Vector2D> &points_0,  // input: 2D image points in the 1st image.
        const std::vector<Vector2D> &points_1,  // input: 2D image points in the 2nd image.
        std::vector<Vector3D> &points_3d,       // output: reconstructed 3D points
        Matrix33 &R,   // output: recovered rotation of the 2nd camera
        Vector3D &t    // output: recovered translation of the 2nd camera
) const
{
    // Check input validity.
    if (points_0.size() < 8 || points_1.size() < 8 || points_0.size() != points_1.size()) {
        std::cerr << "Insufficient or mismatched point correspondences." << std::endl;
        return false;
    }

    // =============================================================================================================
    // #1 Estimate the fundamental matrix F
    // =============================================================================================================
    std::vector<Vector2D> norm_points_0, norm_points_1;
    Matrix33 T0, T1;
    // Use our normalization function (make sure the name matches)
    norm_transformation(points_0, norm_points_0, T0);
    norm_transformation(points_1, norm_points_1, T1);

    Matrix33 F = estimate_fundamental_matrix(norm_points_0, norm_points_1);
    F = T1.transpose() * F * T0;

    // =============================================================================================================
    // #2 Compute the essential matrix E
    // =============================================================================================================
    Matrix33 K(fx, s, cx,
               0, fy, cy,
               0,  0,  1);
    Matrix33 E = compute_essential_matrix(F, K);

    // =============================================================================================================
    // #3 Recover rotation R and translation t from the essential matrix
    // =============================================================================================================
    std::vector<Matrix33> rotations;
    std::vector<Vector3D> translations;
    extract_camera_poses(E, rotations, translations);

    Matrix33 best_R;
    Vector3D best_t;
    int best_index = find_best_pose(rotations, translations, K, points_0, points_1, best_R, best_t);
    R = best_R;
    t = best_t;

    // =============================================================================================================
    // #4 Triangulate 3D points using the recovered camera pose
    // =============================================================================================================
    bool triangulation_success = linear_triangulation(K, R, t, points_0, points_1, points_3d);
    if (!triangulation_success || points_3d.empty()) {
        std::cerr << "Triangulation failed." << std::endl;
        return false;
    }

    // =============================================================================================================
    // #5 Optional: Non-linear refinement.
    // =============================================================================================================
    // Build projection matrices for camera 1 and camera 2.
    // For camera 1, the projection matrix is P0 = K * [I | 0].
    Matrix M0(3, 4, {
        K(0,0), K(0,1), K(0,2), 0,
        K(1,0), K(1,1), K(1,2), 0,
        K(2,0), K(2,1), K(2,2), 0
    });
    // For camera 2, the projection matrix is P1 = K * [R | t].
    Matrix M1(3, 4, {
        (K * R)(0,0), (K * R)(0,1), (K * R)(0,2), t.x(),
        (K * R)(1,0), (K * R)(1,1), (K * R)(1,2), t.y(),
        (K * R)(2,0), (K * R)(2,1), (K * R)(2,2), t.z()
    });

    bool refinement_success = nonlinear_refinement(M0, M1, points_0, points_1, points_3d);
    if (!refinement_success) {
        std::cerr << "Non-linear refinement failed." << std::endl;
        return false;
    }

    double reproj_error = compute_reprojection_error(K, R, t, points_0, points_1, points_3d);
    std::cout << "Average reprojection error: " << reproj_error << std::endl;
    return true;
}

// Make sure this function is declared in the Triangulation header file
static bool solve_linear_system(const Matrix& A, const Vector& b, Vector& x) {
    try {
        Matrix A_inv = inverse(A);
        x = A_inv * b;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error solving linear system: " << e.what() << std::endl;
        return false;
    }
}
