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

#include "calibration.h"
#include "matrix_algo.h"
#include <stdexcept>

using namespace easy3d;

bool equal_matrices(const Matrix &A, const Matrix &B) {
    if (A.cols() == B.cols() && A.rows() == B.rows())
    {
        if ((A - B).resize(1, A.cols()*A.rows()).get_row(0).length() == 0)
        {
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}

double sin_from_cos(double cos_theta, bool positive = true) {
    double sin_theta = std::sqrt(1 - cos_theta * cos_theta);
    return positive ? sin_theta : -sin_theta;
}

double cot_from_cos(double cos_theta, bool positive = true) {
    double cot_theta = cos_theta / sin_from_cos(cos_theta);
    return positive ? cot_theta : -cot_theta;
}

// Free function: Projects 3D points to 2D using the computed camera matrix.
std::vector<Vector2D> project_3D_to_2D(
    const std::vector<Vector3D>& points_3d,
    const Matrix33& K,
    const Matrix33& R,
    const Vector3D& t
) {
    std::vector<Vector2D> projected_points;
    Matrix M(3, 4);
    for (int i = 0; i < 3; i++) {
        M.set_row(i, { R(i, 0), R(i, 1), R(i, 2), t[i] });
    }
    M = K * M;  // M = K * [R | t]
    for (const auto& point_3d : points_3d) {
        Vector4D P_homogeneous = point_3d.homogeneous();
        Vector3D projected = M * P_homogeneous;
        double x = projected.x() / projected.z();
        double y = projected.y() / projected.z();
        projected_points.emplace_back(x, y);
    }
    return projected_points;
}

// Free function: Computes the reprojection error between projected points and the original image points.
std::vector<double> compute_reprojection_error(
    const std::vector<Vector2D>& projected_points,
    const std::vector<Vector2D>& image_points
) {
    assert(projected_points.size() == image_points.size());
    double error = 0.0;
    std::vector<double> error_vec;
    int n = projected_points.size();
    for (int i = 0; i < n; ++i) {
        double dx = projected_points[i].x() - image_points[i].x();
        double dy = projected_points[i].y() - image_points[i].y();
        error = std::sqrt(dx * dx + dy * dy);
        error_vec.push_back(error);
    }
    return error_vec;
}


/**
 * Performs camera calibration given 3D-2D point correspondences.
 * 
 * @param[in] points_3d input: An array of 3D points.
 * @param[in] points_2d input: An array of 2D image points.
 * @param[out] fx output: focal length (i.e., K[0][0]).
 * @param[out] fy output: focal length (i.e., K[1][1]).
 * @param[out] cx output: x component of the principal point (i.e., K[0][2]).
 * @param[out] cy output: y component of the principal point (i.e., K[1][2]).
 * @param[out] s output: skew factor (i.e., K[0][1]), which is s = -alpha * cot(theta).
 * @param[out] R output: the 3x3 rotation matrix encoding camera rotation.
 * @param[out] t output: a 3D vector encoding camera translation.
 * @return True on success, otherwise false. On success, the camera parameters are returned by fx, fy, cx, cy, skew, R, and t).
 */
bool Calibration::calibration(
    const std::vector<Vector3D>& points_3d, 
    const std::vector<Vector2D>& points_2d,
    double& fx,
    double& fy,
    double& cx,
    double& cy,
    double& s,
    Matrix33& R,
    Vector3D& t
) 
{
    //--------------------------------------------------------------------------------------------------------------
    // implementation starts ...
    try {
        // =====================================================================================================================
        // check if input is valid (e.g., number of correspondences >= 6, sizes of 2D/3D points must match)
        // =====================================================================================================================

        int size_2d = points_2d.size();
        int size_3d = points_3d.size();
        if (size_2d < 6) throw std::invalid_argument("number of correspondences must >= 6, got " + std::to_string(size_2d));
        if (size_3d < 6) throw std::invalid_argument("number of correspondences must >= 6, got " + std::to_string(size_3d));
        if (size_2d != size_3d) throw std::invalid_argument("sizes of 2D/3D points must match, got " + std::to_string(size_2d) + " and " + std::to_string(size_3d));

        // =====================================================================================================================
        // construct the P matrix (so P * m = 0).
        // =====================================================================================================================

        int row_num = 2*size_3d;
        Matrix P(row_num, 12, 0.0);
        std::vector<double> tmp;

        for (int i = 0; i < size_3d; i++) {                             // For 3D-2D point correspondence 
            Vector4D P_i = points_3d[i].homogeneous();                  // Transform 3D point P_i into homogeneous coordinates (4D vector)
            Vector4D u_P_i = P_i * points_2d[i].x() * (-1);             // Calculate -(x_i * P_i)
            Vector4D v_P_i = P_i * points_2d[i].y() * (-1);             // Calculate -(y_i * P_i)
            
            tmp.insert(tmp.end(), P_i.data(), P_i.data()+4);            // Add P_i as homogenous coordinates at end of tmp (before empty)
            tmp.insert(tmp.end(), 4, 0);                                // Add four zeroes at end of tmp
            tmp.insert(tmp.end(), u_P_i.data(), u_P_i.data()+4);        // Add u_P_i at end of tmp
            P.set_row(2*i, tmp);                                        // Insert tmp, now 12 values long, into matrix P
            tmp.clear();                                                // Clear tmp of its contents
            tmp.insert(tmp.end(), 4, 0);                                // Add four zeroes at end of tmp (before empty)
            tmp.insert(tmp.end(), P_i.data(), P_i.data()+4);            // Add P_i as homogenous coordinates at end of tmp
            tmp.insert(tmp.end(), v_P_i.data(), v_P_i.data()+4);        // Add v_P_i at end of tmp
            P.set_row(2*i+1, tmp);                                      // Insert tmp, now 12 values long, into matrix P
            tmp.clear();                                                // Clear tmp of its contents
        }
        std::cout << "P: \n" << P << std::endl;

        // =====================================================================================================================
        // solve for M (the whole projection matrix, i.e., M = K * [R, t]) using SVD decomposition.
        // =====================================================================================================================

        Matrix U(row_num, row_num, 0.0);
        Matrix S(row_num, 12, 0.0);
        Matrix V(12, 12, 0.0);
        svd_decompose(P, U, S, V);

        // check if the SVD result is correct
        if (equal_matrices(U * transpose(U), identity(row_num, 1))) throw std::invalid_argument("U * U^T must be identity!");
        if (equal_matrices(V * transpose(V), identity(12, 1))) throw std::invalid_argument("V * V^T must be identity!");
        if (equal_matrices(P, U * S * transpose(V))) throw std::invalid_argument("P != U * S * V^T!");
        
        Matrix m(12, 1, V.get_column(11).data());
        Matrix svd_M(3, 4, V.get_column(11).data());
        Matrix33 A(svd_M);
        Matrix b(3, 1, svd_M.get_column(3).data());

        std::cout << "Is |Pm|^2 close to 0? |Pm|^2:\n" << (P*m).get_column(0).length2() << std::endl << std::endl;

        // =====================================================================================================================
        // extract intrinsic parameters from M.
        // =====================================================================================================================

        Vector3D a1xa3 = cross(A.get_row(0), A.get_row(2));
        Vector3D a2xa3 = cross(A.get_row(1), A.get_row(2));

        int rho_sign = 1;
        if (b[2][0] < 0) {
            rho_sign = -1;
        }

        double rho = rho_sign / A.get_row(2).length();
        cx = std::pow(rho, 2) * dot(A.get_row(0), A.get_row(2));
        cy = std::pow(rho, 2) * dot(A.get_row(1), A.get_row(2));
        double cos_theta = (-1) * dot(a1xa3, a2xa3) / (a1xa3.length() * a2xa3.length());
        fx = std::pow(rho, 2) * a1xa3.length() * sin_from_cos(cos_theta);
        fy = std::pow(rho, 2) * a2xa3.length();
        s = fx * cot_from_cos(cos_theta);

        // =====================================================================================================================
        // extract extrinsic parameters from M.
        // =====================================================================================================================
        
        Matrix K(3, 3, std::vector<double>({fx, s, cx, 0, fy, cy, 0, 0, 1}));
        Vector3D r1 = a2xa3.normalize();
        Vector3D r3 = rho * A.get_row(2);
        Vector3D r2 = cross(r3, r1);
        R.set_row(0, r1);
        R.set_row(1, r2);
        R.set_row(2, r3);

        Matrix invK;
        if (inverse(K, invK)) {
            t = (rho * invK * b).get_column(0);
        } else {
            throw std::runtime_error("expected a square matrix, got " + std::to_string(K.rows()) + " by " + std::to_string(K.cols()));
        }
        std::cout << "The chosen rho: \n" << rho << std::endl <<  std::endl;
        std::cout << "The solved matrix K: \n" << K << std::endl;
        std::cout << "The solved matrix R: \n" << R << std::endl;
        std::cout << "The solved matrix t: \n" << t << std::endl;
        std::cout << "The focal length fx fy: \n" << fx << "\t" << fy << std::endl << std::endl;
        
        // =====================================================================================================================
        // verify the error.
        // =====================================================================================================================

        std::vector<Vector2D> projected_2d = project_3D_to_2D(points_3d, K, R, t);
        auto err_vec = compute_reprojection_error(projected_2d, points_2d);

        std::cout << "the projected\t|  ground truth\t|  error" << std::endl;
        for (size_t i = 0; i < projected_2d.size(); i++) {
            std::cout << projected_2d[i] << "\t|  " << points_2d[i] << "\t|  " << err_vec[i] << std::endl;
        }
        double err_sum = 0.0;
        for (double num: err_vec) {
            err_sum += num;
        }
        std::cout << "the reprojection error (avg): " << err_sum/err_vec.size() << std::endl;

        return true;
    }
    catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
        return false;
    }
}

















