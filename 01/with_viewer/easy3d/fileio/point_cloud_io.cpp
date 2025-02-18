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

#include <easy3d/fileio/point_cloud_io.h>

#include <fstream>

#include <easy3d/util/line_stream.h>
#include <easy3d/util/logging.h>

namespace easy3d {

		bool load_xyz(const std::string& file_name, PointCloud& cloud) {
			std::ifstream input(file_name.c_str());
			if (input.fail()) {
                LOG(ERROR) << "could not open file: " << file_name;
				return false;
			}

			io::LineInputStream in(input);

			vec3 p;
			while (!input.eof()) {
				in.get_line();;
				if (in.current_line()[0] != '#') {
					in >> p;
					if (!in.fail())
						cloud.add_vertex(p);
				}
			}

			return true;
		}


		bool save_xyz(const std::string& file_name, const PointCloud& cloud) {
			std::ofstream output(file_name.c_str());
			if (output.fail()) {
                LOG(ERROR) << "could not open file: " << file_name;
				return false;
			}
			output.precision(16);

			PointCloud::VertexProperty<vec3> points = cloud.get_vertex_property<vec3>("v:point");
            for (auto v : cloud.vertices())
                output << points[v] << std::endl;

			return true;
		}

} // namespace easy3d
