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



double sin_from_cos(double cos_theta, bool positive = true) {
    double sin_theta = std::sqrt(1 - cos_theta * cos_theta);
    return positive ? sin_theta : -sin_theta;
}

double cot_from_cos(double cos_theta, bool positive = true) {
    double cot_theta =  cos_theta / sin_from_cos(cos_theta);
    return positive ? cot_theta : -cot_theta;
}

/**
 * TODO: Finish this function for calibrating a camera from the corresponding 3D-2D point pairs.
 *       You may define a few functions for some sub-tasks.
 * @return True on success, otherwise false. On success, the camera parameters are returned by fx, fy, cx, cy, skew, R, and t).
 */
bool Calibration::calibration(
        const std::vector<Vector3D>& points_3d, /// input: An array of 3D points.
        const std::vector<Vector2D>& points_2d, /// input: An array of 2D image points.
        double& fx,  /// output: focal length (i.e., K[0][0]).
        double& fy,  /// output: focal length (i.e., K[1][1]).
        double& cx,  /// output: x component of the principal point (i.e., K[0][2]).
        double& cy,  /// output: y component of the principal point (i.e., K[1][2]).
        double& s,   /// output: skew factor (i.e., K[0][1]), which is s = -alpha * cot(theta).
        Matrix33& R, /// output: the 3x3 rotation matrix encoding camera rotation.
        Vector3D& t) /// output：a 3D vector encoding camera translation.
{

    std::cout << "\n\tTODO: After you implement this function, please return 'true' - this will trigger the viewer to\n"
                 "\t\tupdate the rendering using your recovered camera parameters. This can help you to visually check\n"
                 "\t\tif your calibration is successful or not.\n\n" << std::flush;

    //--------------------------------------------------------------------------------------------------------------
    // implementation starts ...
    try {
        // check if input is valid (e.g., number of correspondences >= 6, sizes of 2D/3D points must match)

        int size_2d = points_2d.size();
        int size_3d = points_3d.size();
        if (size_2d < 6) throw std::runtime_error("number of correspondences must >= 6, got " + std::to_string(size_2d));
        if (size_3d < 6) throw std::runtime_error("number of correspondences must >= 6, got " + std::to_string(size_3d));
        if (size_2d != size_3d) throw std::runtime_error("sizes of 2D/3D points must match, got " + std::to_string(size_2d) + " and " + std::to_string(size_3d));

        // construct the P matrix (so P * m = 0).
        Matrix P(2*size_3d, 12, 0.0);
        std::vector<double> tmp;

        for (int i = 0; i < size_3d; i++) {
            Vector4D P_i = points_3d[i].homogeneous();
            Vector4D x_i_P_i = P_i * points_2d[i].x() * (-1);
            Vector4D y_i_P_i = P_i * points_2d[i].y() * (-1);
            
            tmp.insert(tmp.end(), P_i.data(), P_i.data()+4);
            tmp.insert(tmp.end(), 4, 0);
            tmp.insert(tmp.end(), x_i_P_i.data(), x_i_P_i.data()+4);
            P.set_row(2*i, tmp);
            tmp.clear();
            tmp.insert(tmp.end(), 4, 0);
            tmp.insert(tmp.end(), P_i.data(), P_i.data()+4);
            tmp.insert(tmp.end(), y_i_P_i.data(), y_i_P_i.data()+4);
            P.set_row(2*i+1, tmp);
            tmp.clear();
        }
        std::cout << "P: \n" << P << std::endl;

        // solve for M (the whole projection matrix, i.e., M = K * [R, t]) using SVD decomposition.
        // Optional: you can check if your M is correct by applying M on the 3D points. If correct, the projected point
        //           should be very close to your input images points.

        Matrix U(2*size_3d, 2*size_3d, 0.0);
        Matrix S(2*size_3d, 12, 0.0);
        Matrix V(12, 12, 0.0);
        svd_decompose(P, U, S, V);

        Matrix svd_M(3, 4, V.get_column(11).data());
        Matrix33 A(svd_M);
        Matrix b(3, 1, svd_M.get_column(3).data());
        // std::cout << "The SVD-solved matrix M: \n" << svd_M << std::endl;
        // std::cout << "The SVD-solved matrix A: \n" << A << std::endl;
        // std::cout << "The SVD-solved matrix b: \n" << b << std::endl;

        // extract intrinsic parameters from M.

        Vector3D a1xa3 = cross(A.get_row(0), A.get_row(2));
        Vector3D a2xa3 = cross(A.get_row(1), A.get_row(2));

        double rho = 1 / A.get_row(2).length();
        cx = std::pow(rho, 2) * dot(A.get_row(0), A.get_row(2));
        cy = std::pow(rho, 2) * dot(A.get_row(1), A.get_row(2));
        double cos_theta = (-1) * dot(a1xa3, a2xa3) / (a1xa3.length() * a2xa3.length());
        fx = std::pow(rho, 2) * a1xa3.length() * sin_from_cos(cos_theta);
        fy = std::pow(rho, 2) * a2xa3.length();
        s = fx * cot_from_cos(cos_theta);

        // extract extrinsic parameters from M.
        
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
        
        return true;
    }
    catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
        return false;
    }
}

















