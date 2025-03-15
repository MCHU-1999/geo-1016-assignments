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

    norm_points.resize(n);  // <--- Fix: Resize before accessing elements
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
        int n0 = points_0.size();
        int n1 = points_1.size();
        if (n0 < 6) throw std::invalid_argument("number of correspondences must >= 8, got " + std::to_string(n0));
        if (n1 < 6) throw std::invalid_argument("number of correspondences must >= 8, got " + std::to_string(n1));
        if (n0 != n1) throw std::invalid_argument("amount of points from both camera must match, got " + std::to_string(n0) + " and " + std::to_string(n1));


        // Estimate relative pose of two views. This can be subdivided into
        // estimate the fundamental matrix F;
        std::vector<Vector2D> norm_points_0, norm_points_1;
        Matrix33 T0, T1;
        norm_transformation(points_0, norm_points_0, T0);
        norm_transformation(points_1, norm_points_1, T1);

        Matrix W = cal_matrix_W(n0, norm_points_0, norm_points_1);
        Matrix U(n0, n0, 0.0), S(n0, 9, 0.0), V(9, 9, 0.0);
        svd_decompose(W, U, S, V);

        Matrix F(3, 3, V.get_column(8).data());
        U = Matrix(3, 3, 0.0);
        V = Matrix(3, 3, 0.0);
        Matrix D(3, 3, 0.0);
        svd_decompose(F, U, D, V);
        D.set(2, 2, 0.0);

        F = U*D*V;
        F = T1.transpose() * F * T0;

        // compute the essential matrix E
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

        // Ensure R has positive determinant (rotation matrix property)
        if (determinant(R1) < 0) R1 = -R1;
        if (determinant(R2) < 0) R2 = -R2;

        // Translation is the last column of U (up to scale)
        Vector3D t1(U_e.get_column(2)[0], U_e.get_column(2)[1], U_e.get_column(2)[2]);
        Vector3D t2 = -t1;

        // Four possible combinations of R and t
        std::vector<Matrix33> rotations = {R1, R1, R2, R2};
        std::vector<Vector3D> translations = {t1, t2, t1, t2};
        std::vector<int> counter = {0, 0, 0, 0};
        for (int i=0; i < 4; i++){
            for (int j = 0; j < n0; j++) {
                Vector3D temp_P = inverse(K) * points_0[i].homogeneous();
                Vector3D Q = rotations[i] * temp_P + translations[i];
                Vector3D temp_Q = inverse(K) * points_1[i].homogeneous();
                Vector3D P = rotations[i].transpose() * (temp_Q - translations[i]);
                if (P[2] > 0 && Q[2] > 0) {
                    counter[i]++;
                }
            }
        }
        auto maxIt = std::max_element(counter.begin(), counter.end());
        int maxIndex = std::distance(counter.begin(), maxIt);
        R = rotations[maxIndex];
        t = translations[maxIndex];
        std::cout << "n= " << n0 << std::endl;
        std::cout << "R1t1 count= " << counter[0] << std::endl;
        std::cout << "R1t2 count= " << counter[1] << std::endl;
        std::cout << "R2t1 count= " << counter[2] << std::endl;
        std::cout << "R2t2 count= " << counter[3] << std::endl;
        std::cout << "Selected R" << maxIndex/2 + 1 << "t" << maxIndex%2 + 1 << std::endl;
        std::cout << "Selected R:\n" << rotations[maxIndex] << std::endl;
        std::cout << "Selected t:\n" << translations[maxIndex] << std::endl << std::endl;

        // Reconstruct 3D points. The main task is
        // triangulate a pair of image points (i.e., compute the 3D coordinates for each corresponding point pair)
        Matrix KR = K*R;
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

        Matrix A(4, 4);
        U = Matrix(4, 4, 0.0);
        S = Matrix(4, 4, 0.0);
        V = Matrix(4, 4, 0.0);
        for (int i = 0; i < n0; i++) {
            A.set_row(0, points_0[i].x() * M0.get_row(2) - M0.get_row(0));
            A.set_row(1, points_0[i].y() * M0.get_row(2) - M0.get_row(1));
            A.set_row(2, points_1[i].x() * M1.get_row(2) - M1.get_row(0));
            A.set_row(3, points_1[i].y() * M1.get_row(2) - M1.get_row(1));
            
            svd_decompose(A, U, S, V);
            Vector4D P_homo = V.get_column(3);
            points_3d.push_back(P_homo.cartesian());
            // std::cout << "P= " << P_homo.cartesian() << std::endl;
        }

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