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

    for (int i = 0; i < n; i++) {
        norm_points[i] = (scale * points[i]) - norm_centroid;
    }
}

Matrix cal_matrix_W (int n, const std::vector<Vector2D> &points_0, const std::vector<Vector2D> &points_1) {
    Matrix W(n, 9);
    for (int i = 0; i < n; i++) {
        // u_i  = norm_points_0[i].x()
        // v_i  = norm_points_0[i].y()
        // u_i' = norm_points_1[i].x()
        // v_i' = norm_points_1[i].y()
        // Row of W:
        // u_i'u_i, u_i'v_i, u_i'
        // v_i'u_i, v_i'v_i, v_i'
        // u_i    , v_i    , 1
        double u_0 = points_0[i].x(), v_0 = points_0[i].y(), u_1 = points_1[i].x(), v_1 = points_1[i].y();
        W.set_row(i, {
            u_1*u_0, u_1*v_0, u_1,
            v_1*u_0, v_1*v_0, v_1,
            u_0, v_0, 1
        });
    }

    return W;
}


/**
 * TODO: Finish this function for reconstructing 3D geometry from corresponding image points.
 * @return True on success, otherwise false. On success, the reconstructed 3D points must be written to 'points_3d'
 *      and the recovered relative pose must be written to R and t.
 */
bool Triangulation::triangulation(
        double fx, double fy,     /// input: the focal lengths (same for both cameras)
        double cx, double cy,     /// input: the principal point (same for both cameras)
        double s,                 /// input: the skew factor (same for both cameras)
        const std::vector<Vector2D> &points_0,  /// input: 2D image points in the 1st image.
        const std::vector<Vector2D> &points_1,  /// input: 2D image points in the 2nd image.
        std::vector<Vector3D> &points_3d,       /// output: reconstructed 3D points
        Matrix33 &R,   /// output: 3 by 3 matrix, which is the recovered rotation of the 2nd camera
        Vector3D &t    /// output: 3D vector, which is the recovered translation of the 2nd camera
) const
{
    //--------------------------------------------------------------------------------------------------------------
    // implementation starts ...

    try {
        // check if the input is valid (always good because you never known how others will call your function).
        int n_0 = points_0.size();
        int n_1 = points_1.size();
        if (n_0 < 6) throw std::invalid_argument("number of correspondences must >= 8, got " + std::to_string(n_0));
        if (n_1 < 6) throw std::invalid_argument("number of correspondences must >= 8, got " + std::to_string(n_1));
        if (n_0 != n_1) throw std::invalid_argument("amount of points from both camera must match, got " + std::to_string(n_0) + " and " + std::to_string(n_1));


        // Estimate relative pose of two views. This can be subdivided into
        // estimate the fundamental matrix F;
        std::vector<Vector2D> norm_points_0, norm_points_1;
        Matrix33 T_0, T_1;
        norm_transformation(points_0, norm_points_0, T_0);
        norm_transformation(points_1, norm_points_1, T_1);

        Matrix W = cal_matrix_W(n_0, norm_points_0, norm_points_1);
        Matrix U(n_0, n_0, 0.0), S(n_0, 9, 0.0), V(9, 9, 0.0);
        svd_decompose(W, U, S, V);
        Matrix F(3, 3, V.get_column(8).data());
        U = Matrix(3, 3, 0.0);
        V = Matrix(3, 3, 0.0);
        Matrix D(3, 3, 0.0);
        svd_decompose(F, U, S, V);
        S.set(2, 2, 0.0);
        F = U*S*V;
        F = T_1.transpose() * F * T_0;

        // compute the essential matrix E;
        Matrix33 K({fx, s, cx, 0, fy, cy, 0, 0, 1});
        Matrix33 E = K.transpose() * F * K;

        // recover rotation R and t.
        Matrix U_e(3, 3, 0.0), V_e(3, 3, 0.0);
        Matrix S_e(3, 3, 0.0);
        svd_decompose(E, U_e, S_e, V_e);

        // Define the W matrix
        Matrix33 W_matrix({
            0, -1, 0,
            1, 0, 0,
            0, 0, 1
        });

        // Four possible solutions for R and t
        Matrix33 R1 = U_e * W_matrix * V_e.transpose();
        Matrix33 R2 = U_e * W_matrix.transpose() * V_e.transpose();

        std::cout << R1 << std::endl;
        std::cout << R2 << std::endl;

        // Ensure R has positive determinant (rotation matrix property)
        if (determinant(R1) < 0) R1 = -R1;
        if (determinant(R2) < 0) R2 = -R2;

        std::cout << R1 << std::endl;
        std::cout << R2 << std::endl;

        // Translation is the last column of U (up to scale)
        Vector3D t1(U_e.get_column(2)[0], U_e.get_column(2)[1], U_e.get_column(2)[2]);
        Vector3D t2 = -t1;

        // Four possible combinations of R and t
        std::vector<Matrix33> rotations = {R1, R1, R2, R2};
        std::vector<Vector3D> translations = {t1, t2, t1, t2};

        // First camera projection matrix (identity rotation, zero translation)
        Matrix34 P0({
            fx, 0, cx, 0,
            0, fy, cy, 0,
            0, 0, 1, 0
        });





        // TODO: Reconstruct 3D points. The main task is
        //      - triangulate a pair of image points (i.e., compute the 3D coordinates for each corresponding point pair)

        // TODO: Don't forget to
        //          - write your recovered 3D points into 'points_3d' (so the viewer can visualize the 3D points for you);
        //          - write the recovered relative pose into R and t (the view will be updated as seen from the 2nd camera,
        //            which can help you check if R and t are correct).
        //       You must return either 'true' or 'false' to indicate whether the triangulation was successful (so the
        //       viewer will be notified to visualize the 3D points and update the view).
        //       There are a few cases you should return 'false' instead, for example:
        //          - function not implemented yet;
        //          - input not valid (e.g., not enough points, point numbers don't match);
        //          - encountered failure in any step.
        return points_3d.size() > 0;

    }
    catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
        return false;
    }
}