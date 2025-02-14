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

#include "matrix.h"
#include "vector.h"

using namespace easy3d;


int main(int argc, char *argv[]) {
    /// Here are the camera intrinsic parameters
    const double fx = 1000;
    const double fy = 1000;
    const double cx = 320;
    const double cy = 240;

    /// Let's assume an ideal camera (i.e., no skew distortion)
    double skew = 0.0;

    /// Here are a set of 3D object points stored in an array
    std::vector<Vector3D> points_3d = {
            {-0.348852,    0.175479,    1.73528},
            {0.209258,     0.174207,    1.73433},
            {0.243879,     0.167286,    1.71709},
            {0.277675,     0.170791,    1.72754},
            {0.347348,     -0.0377202,  1.72291},
            {0.344641,     -0.00252763, 1.72196},
            {-0.104009,    -0.175297,   1.73351},
            {0.24819,      -0.173531,   1.74182},
            {0.277353,     -0.174265,   1.73297},
            {0.312613,     -0.175822,   1.72982},
            {0.351051,     -0.175369,   1.73506},
            {-0.0342135,   0.175079,    2.084},
            {-0.000979426, 0.169598,    2.07162},
            {0.0376033,    0.175466,    2.08317},
            {-0.348618,    -0.141933,   2.08796},
            {-0.348956,    -0.107041,   2.10335},
            {-0.348596,    -0.0679049,  2.09436},
            {0.353343,     -0.0715576,  2.08994},
            {0.346852,     -0.038986,   2.06997},
            {0.348411,     -0.00237281, 2.07406},
            {-0.2101,      -0.175002,   2.08682},
            {-0.176409,    -0.177025,   2.0795},
            {0.310176,     -0.176414,   2.07904},
            {-0.347476,    0.178684,    1.94968},
            {-0.350333,    0.17386,     1.98595},
            {-0.347188,    0.171499,    2.00665},
            {0.346637,     0.172316,    2.00685},
            {0.344322,     0.172559,    2.03319},
            {-0.348979,    -0.175614,   1.76529},
            {-0.345604,    -0.174258,   2.06471},
            {0.352228,     -0.174178,   1.77861},
            {0.353392,     -0.175458,   2.06091}
    };

    /// Here are the camera extrinsic parameters
    /// Extrinsic parameters - rotation
    Matrix33 R(
            0.965151, -0.00207759, -0.261687,
            -0.066138, 0.96557, -0.251595,
            0.2532, 0.260135, 0.931783
    );
    /// Extrinsic parameters - translation
    Vector3D t(0.676348, 0.630973, 0.380036);

    /// Let's build the 3 by 3 intrinsic matrix
    Matrix33 K(fx, skew, cx,
               0, fy, cy,
               0, 0, 1);

    /// Loop over all 3D points and project each point onto the image plane
    for (std::size_t i = 0; i < points_3d.size(); ++i) {
        const Vector3D &P = points_3d[i];
        /// This is how a 3D point is projected onto the image plane
        Vector3D proj = K * (R * P + t); /// this is in the homogeneous coordinate
        Vector2D p = proj.cartesian(); /// convert it to Cartesian coordinate
        std::cout << "(" << P << ") -> (" << p << ")" << std::endl;
    }

    return EXIT_SUCCESS;
}