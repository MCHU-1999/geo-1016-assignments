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

using namespace easy3d;

/**
 * Normalize 2D points using normalization transformation
 * 
 * @param points Input 2D points
 * @param norm_points Output normalized points
 * @param T Output normalization transformation matrix
 */
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
        
        return !points_3d.empty();
    }
    catch (const std::exception& e) {
        std::cerr << "Triangulation error: " << e.what() << std::endl;
        return false;
    }
}