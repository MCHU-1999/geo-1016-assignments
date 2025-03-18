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

void norm_transformation(const std::vector<Vector2D>& points, std::vector<Vector2D>& norm_points, Matrix33& T) {
    // Calculate centroid
    Vector2D centroid(0, 0);
    int n = points.size();
    for (const auto& p : points) {
        centroid += p;
    }
    centroid /= n;

    // Calculate average distance to centroid
    double dis = 0.0;
    for (const auto& p : points) {
        dis += (centroid - p).length();
    }
    double scale = sqrt(2) / (dis / n);
    Vector2D norm_centroid = scale * centroid;

    // Create normalization transformation matrix
    T = Matrix33(
        scale, 0, -norm_centroid.x(),
        0, scale, -norm_centroid.y(),
        0, 0, 1
    );

    // Apply normalization to all points
    norm_points.clear();
    for (int i = 0; i < n; i++) {
        norm_points.push_back((scale * points[i]) - norm_centroid);
    }
}

/**
 * Calculate the W matrix for fundamental matrix estimation
 *
 * @param n Number of point correspondences
 * @param points_0 2D points from first image
 * @param points_1 2D points from second image
 * @return Matrix W for use in SVD
 */
Matrix cal_matrix_W(int n, const std::vector<Vector2D>& points_0, const std::vector<Vector2D>& points_1) {
    Matrix W(n, 9);
    for (int i = 0; i < n; i++) {
        double u0 = points_0[i].x(), v0 = points_0[i].y();
        double u1 = points_1[i].x(), v1 = points_1[i].y();
        W.set_row(i, {
            u1 * u0, u1 * v0, u1,
            v1 * u0, v1 * v0, v1,
            u0, v0, 1
        });
    }
    return W;
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
    const int max_iterations = 1000;  // Maximum number of iterations
    const double epsilon = 1e-6;    // Convergence threshold

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
    Matrix34 P0 = Matrix34({
        K(0,0), K(0,1), K(0,2), 0,
        K(1,0), K(1,1), K(1,2), 0,
        K(2,0), K(2,1), K(2,2), 0
    });

    // P1 = K * [R | t]
    Matrix34 Rt({
        R(0,0), R(0,1), R(0,2), t.x(),
        R(1,0), R(1,1), R(1,2), t.y(),
        R(2,0), R(2,1), R(2,2), t.z()
    });
    Matrix34 P1 = K * Rt;

    for (int i = 0; i < n; ++i) {
        // Create homogeneous point
        Vector4D P_homo(points_3d[i].x(), points_3d[i].y(), points_3d[i].z(), 1.0);

        // Project to first image
        Vector3D proj0 = P0 * P_homo;
        Vector2D p0_proj(proj0.x() / proj0.z(), proj0.y() / proj0.z());

        // Project to second image
        Vector3D proj1 = P1 * P_homo;
        Vector2D p1_proj(proj1.x() / proj1.z(), proj1.y() / proj1.z());

        /// Calculate reprojection error (squared Euclidean distance)
        double dx0 = points_0[i].x() - p0_proj.x();
        double dy0 = points_0[i].y() - p0_proj.y();
        double error0 = dx0 * dx0 + dy0 * dy0;  // Squared error for the first image

        double dx1 = points_1[i].x() - p1_proj.x();
        double dy1 = points_1[i].y() - p1_proj.y();
        double error1 = dx1 * dx1 + dy1 * dy1;  // Squared error for the second image

        // Add squared errors
        total_error += error0 + error1;
    }

    // Return average reprojection error
    return total_error / (2.0 * n);  // Average error per point per image
}

/**
 * @brief Performs 3D point triangulation from two images.
 *
 * Given two sets of 2D point correspondences from two images and known camera intrinsics,
 * this function reconstructs the 3D positions of the points and estimates the relative
 * pose (rotation and translation) between the two cameras.
 *
 * @param[in] fx Focal length in the x direction (same for both cameras).
 * @param[in] fy Focal length in the y direction (same for both cameras).
 * @param[in] cx X-coordinate of the principal point (same for both cameras).
 * @param[in] cy Y-coordinate of the principal point (same for both cameras).
 * @param[in] s Skew factor (same for both cameras).
 * @param[in] points_0 2D image points in the first image.
 * @param[in] points_1 2D image points in the second image.
 * @param[out] points_3d Reconstructed 3D points in world coordinates.
 * @param[out] R 3×3 matrix representing the recovered rotation of the 2nd camera.
 * @param[out] t 3D vector representing the recovered translation of the 2nd camera.
 * @return True on success, otherwise false.
 */
bool Triangulation::triangulation(
    double fx, double fy,
    double cx, double cy,
    double s,
    const std::vector<Vector2D>& points_0,
    const std::vector<Vector2D>& points_1,
    std::vector<Vector3D>& points_3d,
    Matrix33& R,
    Vector3D& t
) const {
    try {
        // 1. Validate input
        int n0 = points_0.size();
        int n1 = points_1.size();
        if (n0 < 8) throw std::invalid_argument("Number of correspondences must be >= 8, got " + std::to_string(n0));
        if (n1 < 8) throw std::invalid_argument("Number of correspondences must be >= 8, got " + std::to_string(n1));
        if (n0 != n1) throw std::invalid_argument("Amount of points from both cameras must match, got " + std::to_string(n0) + " and " + std::to_string(n1));

        // Clear output vector to ensure we start fresh
        points_3d.clear();

        // 2. Normalize points for numerical stability
        std::vector<Vector2D> norm_points_0, norm_points_1;
        Matrix33 T0, T1;
        norm_transformation(points_0, norm_points_0, T0);
        norm_transformation(points_1, norm_points_1, T1);

        // 3. Calculate fundamental matrix using 8-point algorithm
        Matrix W = cal_matrix_W(n0, norm_points_0, norm_points_1);
        Matrix U(n0, n0), S(n0, 9), V(9, 9);
        svd_decompose(W, U, S, V);

        // Get the last column of V as the solution (corresponds to smallest singular value)
        Matrix F(3, 3, V.get_column(8).data());

        // 4. Enforce rank-2 constraint on F
        Matrix U_f(3, 3), D_f(3, 3), V_f(3, 3);
        svd_decompose(F, U_f, D_f, V_f);
        D_f.set(2, 2, 0.0);  // Set smallest singular value to zero

        // Recompute F with rank-2 constraint
        F = U_f * D_f * V_f.transpose();

        // 5. Denormalize F
        F = T1.transpose() * F * T0;

        // 6. Compute essential matrix from fundamental matrix
        Matrix33 K({
            fx, s, cx,
            0, fy, cy,
            0, 0, 1
        });
        Matrix33 E = K.transpose() * F * K;

        // 7. Decompose essential matrix to extract rotation and translation
        Matrix U_e(3, 3), S_e(3, 3), V_e(3, 3);
        svd_decompose(E, U_e, S_e, V_e);

        // Define the W matrix for rotation extraction
        Matrix33 W_matrix({
            0, -1, 0,
            1, 0, 0,
            0, 0, 1
        });

        // Four possible solutions for R and t
        Matrix33 R1 = U_e * W_matrix * V_e.transpose();
        Matrix33 R2 = U_e * W_matrix.transpose() * V_e.transpose();

        // Ensure R has positive determinant (rotation matrix property)
        if (determinant(R1) < 0) R1 = -R1;
        if (determinant(R2) < 0) R2 = -R2;

        // Translation is the last column of U_e (up to scale)
        Vector3D t1 = U_e.get_column(2);
        Vector3D t2 = -t1;

        // 8. Set up camera matrices for triangulation
        std::vector<Matrix33> rotations = {R1, R1, R2, R2};
        std::vector<Vector3D> translations = {t1, t2, t1, t2};

        // 9. Select correct rotation and translation using cheirality check
        std::vector<int> counter(4, 0);
        std::vector<std::vector<Vector3D>> all_points_3d(4);

        for (int config = 0; config < 4; config++) {
            Matrix34 I0({   // First camera at origin [I|0]
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0
            });
            Matrix34 Rt({   // Second camera [R|t]
                rotations[config][0][0], rotations[config][0][1], rotations[config][0][2], translations[config].x(),
                rotations[config][1][0], rotations[config][1][1], rotations[config][1][2], translations[config].y(),
                rotations[config][2][0], rotations[config][2][1], rotations[config][2][2], translations[config].z()
            });
            Matrix34 M0 = K*I0;
            Matrix34 M1 = K*Rt;

            // Triangulate points with current camera configuration
            for (int i = 0; i < n0; i++) {
                Matrix A(4, 4);
                A.set_row(0, points_0[i].x() * M0.get_row(2) - M0.get_row(0));
                A.set_row(1, points_0[i].y() * M0.get_row(2) - M0.get_row(1));
                A.set_row(2, points_1[i].x() * M1.get_row(2) - M1.get_row(0));
                A.set_row(3, points_1[i].y() * M1.get_row(2) - M1.get_row(1));

                // Solve using SVD
                Matrix U_tri(4, 4), S_tri(4, 4), V_tri(4, 4);
                svd_decompose(A, U_tri, S_tri, V_tri);

                // Get homogeneous point from last column of V
                Vector4D P_homo = V_tri.get_column(3);
                Vector3D P = P_homo.cartesian();
                Vector3D Q = rotations[config] * P + translations[config];
                all_points_3d[config].push_back(P);

                // Check if point is in front of both cameras
                if (P.z() > 0 && Q.z() > 0) {
                    counter[config]++;
                }
            }
        }

        // Find configuration with most points in front of both cameras
        auto maxIt = std::max_element(counter.begin(), counter.end());
        int best_config = std::distance(counter.begin(), maxIt);

        std::cout << "Points in front of both cameras:" << std::endl;
        std::cout << "R1t1: " << counter[0] << ", R1t2: " << counter[1] << std::endl;
        std::cout << "R2t1: " << counter[2] << ", R2t2: " << counter[3] << std::endl;
        std::cout << "Selected configuration: R" << (best_config/2 + 1) << "t" << (best_config%2 + 1) << std::endl;

        // 10. Set output rotation and translation
        R = rotations[best_config];
        t = translations[best_config];

        // 11. Use triangulated points from the best configuration
        points_3d = all_points_3d[best_config];

        // =============================================================================================================
        // // #12 Non-linear refinement of triangulated points
        // // =============================================================================================================
        // // Build projection matrices for camera 1 and camera 2.
        // Matrix34 M0({   // First camera at origin [I|0]
        //     1, 0, 0, 0,
        //     0, 1, 0, 0,
        //     0, 0, 1, 0
        // });
        // Matrix34 Rt({   // Second camera [R|t]
        //     R[0][0], R[0][1], R[0][2], t.x(),
        //     R[1][0], R[1][1], R[1][2], t.y(),
        //     R[2][0], R[2][1], R[2][2], t.z()
        // });
        // Matrix34 M1 = K * Rt;
        //
        // // Perform non-linear refinement
        // bool refinement_success = nonlinear_refinement(M0, M1, points_0, points_1, points_3d);
        // if (!refinement_success) {
        //     std::cerr << "Non-linear refinement failed." << std::endl;
        //     return false;
        // }

        // =============================================================================================================
        // #13 Calculate reprojection error
        // =============================================================================================================
        double reproj_error = compute_reprojection_error(K, R, t, points_0, points_1, points_3d);
        std::cout << "Average reprojection error: " << reproj_error << std::endl;

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Triangulation error: " << e.what() << std::endl;
        return false;
    }
}




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
