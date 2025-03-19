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
#include <cmath>

using namespace easy3d;

/**
 * @brief Solves a linear system Ax = b.
 * @param A Coefficient matrix.
 * @param b Right-hand side vector.
 * @param x Solution vector.
 * @return True if the system is solvable.
 */
static bool solve_linear_system(const Matrix& A, const Vector& b, Vector& x);

/**
 * @brief Normalizes 2D points for numerical stability.
 * @param points Input 2D points.
 * @param norm_points Output normalized points.
 * @param T Normalization transformation matrix.
 */
void norm_transformation(const std::vector<Vector2D>& points, std::vector<Vector2D>& norm_points, Matrix33& T);

/**
 * @brief Computes the W matrix for fundamental matrix estimation.
 * @param n Number of point correspondences.
 * @param points_0 2D points from the first image.
 * @param points_1 2D points from the second image.
 * @return The computed W matrix.
 */
Matrix cal_matrix_W(int n, const std::vector<Vector2D>& points_0, const std::vector<Vector2D>& points_1);

/**
 * @brief Computes the fundamental matrix using the normalized 8-point algorithm.
 * @param points_0 2D points from the first image.
 * @param points_1 2D points from the second image.
 * @return The computed fundamental matrix.
 */
Matrix cal_fundamental(const std::vector<Vector2D>& points_0, const std::vector<Vector2D>& points_1);

/**
 * @brief Performs linear triangulation of 3D points.
 * @param K Camera intrinsic matrix.
 * @param R Camera rotation matrix.
 * @param t Camera translation vector.
 * @param rotations Possible rotation matrices.
 * @param translations Possible translation vectors.
 * @param points_0 2D points from the first image.
 * @param points_1 2D points from the second image.
 * @param Triangulated_3d Output triangulated 3D points.
 * @return True if triangulation is successful.
 */
bool linear_triangulation(
    const Matrix33& K, Matrix33& R, Vector3D& t,
    const std::vector<Matrix33>& rotations, const std::vector<Vector3D>& translations,
    const std::vector<Vector2D>& points_0, const std::vector<Vector2D>& points_1,
    std::vector<Vector3D>& Triangulated_3d
);

/**
 * @brief Computes the reprojection error for triangulated points.
 * @param K Camera intrinsic matrix.
 * @param R Camera rotation matrix.
 * @param t Camera translation vector.
 * @param points_0 Original 2D points from the first image.
 * @param points_1 Original 2D points from the second image.
 * @param points_3d Triangulated 3D points.
 * @return The average reprojection error.
 */
double cal_reprojection_error(
    const Matrix33& K, const Matrix33& R, const Vector3D& t,
    const std::vector<Vector2D>& points_0, const std::vector<Vector2D>& points_1,
    const std::vector<Vector3D>& points_3d
);

/**
 * Refine a 3D point using Levenberg-Marquardt optimization
 * @param P1 Projection matrix for first camera
 * @param P2 Projection matrix for second camera
 * @param point_0 Observed 2D point in first image
 * @param point_1 Observed 2D point in second image
 * @param point_3d Initial 3D point to be refined
 * @return true if optimization succeeded
 */
bool refine_point(
    const Matrix34& P1, const Matrix34& P2,
    const Vector2D& point_0, const Vector2D& point_1,
    Vector3D& point_3d
);

// To use the Levenberg-Marquardt method to solve a non-linear least squares method, 
// we need to define our own function that inherits 'Objective_LM'.
class TriangulationObjective : public Objective_LM {
private:
    Matrix34 P1_;       // Projection matrix for first camera
    Matrix34 P2_;       // Projection matrix for second camera
    Vector2D point_0_;  // Observed 2D point in first image
    Vector2D point_1_;  // Observed 2D point in second image
public:
    TriangulationObjective(
        const Matrix34& P1, const Matrix34& P2,
        const Vector2D& point_0, const Vector2D& point_1
    ) : Objective_LM(4, 3), // 4 functions (x, y for each view), 3 variables (X,Y,Z)
        P1_(P1), P2_(P2), point_0_(point_0), point_1_(point_1) {}

    /**
     *  Calculate the values of each function at x and return the function values as a vector in fvec.
     *  @param  x       The current values of variables.
     *  @param  fvec    Return the value vector of all the functions.
     *  @return Return a negative value to terminate.
     */
    int evaluate(const double *x, double *fvec) override {
        // Create homogeneous 3D point
        Vector4D point_homo(x[0], x[1], x[2], 1.0);

        // Project to first image
        Vector3D proj1 = P1_ * point_homo;
        Vector2D p1_proj(proj1.x() / proj1.z(), proj1.y() / proj1.z());

        // Project to second image
        Vector3D proj2 = P2_ * point_homo;
        Vector2D p2_proj(proj2.x() / proj2.z(), proj2.y() / proj2.z());

        // Compute reprojection errors (x,y for each camera)
        fvec[0] = point_0_.x() - p1_proj.x();
        fvec[1] = point_0_.y() - p1_proj.y();
        fvec[2] = point_1_.x() - p2_proj.x();
        fvec[3] = point_1_.y() - p2_proj.y();

        return 0;
    }
};

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
        // =============================================================================================================
        // Validate input
        // =============================================================================================================
        points_3d.clear();
        int n0 = points_0.size();
        int n1 = points_1.size();
        if (n0 < 8) throw std::invalid_argument("Number of correspondences must be >= 8, got " + std::to_string(n0));
        if (n1 < 8) throw std::invalid_argument("Number of correspondences must be >= 8, got " + std::to_string(n1));
        if (n0 != n1) throw std::invalid_argument("Amount of points from both cameras must match, got " + std::to_string(n0) + " and " + std::to_string(n1));

        // =============================================================================================================
        // Compute fundamental metrix F
        // =============================================================================================================
        Matrix F = cal_fundamental(points_0, points_1);

        // =============================================================================================================
        // Compute essential matrix from fundamental matrix
        // =============================================================================================================
        Matrix33 K({
            fx, s, cx,
            0, fy, cy,
            0,  0,  1
        });
        Matrix33 E = K.transpose() * F * K;

        // =============================================================================================================
        // Decompose essential matrix to extract rotation and translation
        // =============================================================================================================
        Matrix U_e(3, 3), S_e(3, 3), V_e(3, 3);
        svd_decompose(E, U_e, S_e, V_e);

        // Define the W matrix for rotation extraction
        Matrix33 W_matrix({
            0, -1, 0,
            1, 0, 0,
            0, 0, 1
        });

        // Recover R
        Matrix33 R1 = U_e * W_matrix * V_e.transpose();
        Matrix33 R2 = U_e * W_matrix.transpose() * V_e.transpose();
        if (determinant(R1) < 0) R1 = -R1;
        if (determinant(R2) < 0) R2 = -R2;

        // Recover t
        Vector3D t1 = U_e.get_column(2);
        Vector3D t2 = -t1;

        // Set up combinations
        std::vector<Matrix33> rotations = {R1, R1, R2, R2};
        std::vector<Vector3D> translations = {t1, t2, t1, t2};
        
        // =============================================================================================================
        // Select correct R, t and do the triangulation
        // =============================================================================================================
        bool success = linear_triangulation(K, R, t, rotations, translations, points_0, points_1, points_3d);
        if (!success) return false;
        
        // =============================================================================================================
        // Calculate reprojection error (1)
        // =============================================================================================================
        double reproj_error_before = cal_reprojection_error(K, R, t, points_0, points_1, points_3d);

        // =============================================================================================================
        // Non-linear refinement of triangulated points
        // =============================================================================================================
        Matrix34 I0({   // First camera at origin [I|0]
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0
        });
        Matrix34 Rt({   // Second camera [R|t]
            R[0][0], R[0][1], R[0][2], t.x(),
            R[1][0], R[1][1], R[1][2], t.y(),
            R[2][0], R[2][1], R[2][2], t.z()
        });
        Matrix34 M0 = K * I0;
        Matrix34 M1 = K * Rt;

        // Perform non-linear refinement
        int num_refined = 0;
        for (int i = 0; i < points_3d.size(); ++i) {
            bool status = refine_point(M0, M1, points_0[i], points_1[i], points_3d[i]);
            if (status) num_refined++;
        }
        std::cout << "Refined " << num_refined << " out of " << points_3d.size() << " points" << std::endl;
        
        // =============================================================================================================
        // Calculate reprojection error (2)
        // =============================================================================================================
        double reproj_error_after = cal_reprojection_error(K, R, t, points_0, points_1, points_3d);
        std::cout << "\nBefore nonlinear refinement:" << std::endl;
        std::cout << "---------------------------------------------" << std::endl;
        std::cout << "Reprojection error: " << reproj_error_before << std::endl;
        std::cout << "Avg. reprojection error: " << reproj_error_before/n0 << std::endl;
        std::cout << "\nAfter nonlinear refinement:" << std::endl;
        std::cout << "---------------------------------------------" << std::endl;
        std::cout << "Reprojection error: " << reproj_error_after << std::endl;
        std::cout << "Avg. reprojection error: " << reproj_error_after/n0 << "\n" << std::endl;

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
        std::cerr << "Triangulation error: " << e.what() << std::endl;
        return false;
    }
}

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

    // Apply normalization to points
    norm_points.clear();
    for (int i = 0; i < n; i++) {
        norm_points.push_back((scale * points[i]) - norm_centroid);
    }
}

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

Matrix cal_fundamental(const std::vector<Vector2D>& points_0, const std::vector<Vector2D>& points_1) {
    int n = points_0.size();

    // Normalize points for numerical stability
    std::vector<Vector2D> norm_points_0, norm_points_1;
    Matrix33 T0, T1;
    norm_transformation(points_0, norm_points_0, T0);
    norm_transformation(points_1, norm_points_1, T1);

    // Calculate fundamental matrix using 8-point algorithm
    Matrix W = cal_matrix_W(n, norm_points_0, norm_points_1);
    Matrix U(n, n), S(n, 9), V(9, 9);
    svd_decompose(W, U, S, V);

    // Get the last column of V as the solution
    Matrix F(3, 3, V.get_column(8).data());

    // Enforce rank-2 constraint on F
    Matrix U_f(3, 3), D_f(3, 3), V_f(3, 3);
    svd_decompose(F, U_f, D_f, V_f);
    D_f.set(2, 2, 0.0);

    // Recompute F with rank-2 constraint
    F = U_f * D_f * V_f.transpose();

    // Denormalize F and return
    return T1.transpose() * F * T0;
}

bool linear_triangulation(
    const Matrix33& K, Matrix33& R, Vector3D& t,
    const std::vector<Matrix33>& rotations, const std::vector<Vector3D>& translations,
    const std::vector<Vector2D>& points_0, const std::vector<Vector2D>& points_1,
    std::vector<Vector3D>& Triangulated_3d
)
{
    try {
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
            for (int i = 0; i < points_0.size(); i++) {
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
    
        // Get the best configuration
        Triangulated_3d = all_points_3d[best_config];
        R = rotations[best_config];
        t = translations[best_config];
        return true;
    }
    catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
        return false;
    }
}

double cal_reprojection_error(
    const Matrix33& K, const Matrix33& R, const Vector3D& t,
    const std::vector<Vector2D>& points_0, const std::vector<Vector2D>& points_1,
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

    return total_error;
}

bool refine_point(
    const Matrix34& P1, const Matrix34& P2,
    const Vector2D& point_0, const Vector2D& point_1,
    Vector3D& point_3d
) {
    // Create the objective function, optimizer and parameters
    TriangulationObjective obj(P1, P2, point_0, point_1);
    Optimizer_LM lm;
    Optimizer_LM::Parameters params;

    // The default constructor already initializes with reasonable defaults, but we can customize as needed:
    params.maxcall = 500;       // Maximum iterations
    params.epsilon = 1.e-10;    // step used to calculate the Jacobian.
    params.ftol = 1.0e-14;      // Custom function tolerance
    params.xtol = 1.e-14;
    params.gtol = 1.e-14;
    params.stepbound = 500.0;   // Different step bound
    params.nprint = 0;

    // Initialize with current point estimate
    std::vector<double> x = {point_3d.x(), point_3d.y(), point_3d.z()};

    // Optimize to minimize reprojection error
    bool status = lm.optimize(&obj, x, &params);
    if (status) point_3d = Vector3D(x[0], x[1], x[2]);

    return status;
}