/*
*	Copyright (C) 2015 by Liangliang Nan (liangliang.nan@gmail.com)
*	https://3d.bk.tudelft.nl/liangliang/
*
*	This file is part of Easy3D: software for processing and rendering
*   meshes and point clouds.
*
*	Easy3D is free software; you can redistribute it and/or modify
*	it under the terms of the GNU General Public License Version 3
*	as published by the Free Software Foundation.
*
*	Easy3D is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program. If not, see <http://www.gnu.org/licenses/>.
*/


#include <easy3d/core/kdtree.h>
#include <easy3d/core/point_cloud.h>

#include <nanoflann/nanoflann.hpp>


using namespace nanoflann;

namespace easy3d {

    struct PointSet {
        PointSet(const std::vector<vec3>* points) : pts(points) {}
        const std::vector<vec3>*  pts;

        // Must return the number of data points
        inline size_t kdtree_get_point_count() const { return pts->size(); }

        // Returns the dim'th component of the idx'th point in the class:
        // Since this is inlined and the "dim" argument is typically an immediate value, the
        //  "if/else's" are actually solved at compile time.
        inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
            if (dim == 0) return pts->at(idx).x;
            else if (dim == 1) return pts->at(idx).y;
            else return pts->at(idx).z;
        }

        // Optional bounding-box computation: return false to default to a standard bbox computation loop.
        // Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
        // Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
        template <class BBOX>
        bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
    };


    struct KD_Tree : public KDTreeSingleIndexAdaptor< L2_Simple_Adaptor<float, PointSet>, PointSet, 3 > {
        KD_Tree(PointSet* pset) : KDTreeSingleIndexAdaptor< L2_Simple_Adaptor<float, PointSet>, PointSet, 3 >(3, *pset, KDTreeSingleIndexAdaptorParams(10)){
            pset_ = pset;
        }
        ~KD_Tree() { delete pset_; }

        PointSet* pset_;
    };

    #define get_tree(x) (reinterpret_cast<KD_Tree*>(x))



    KdTree::KdTree() {
        points_ = nullptr;
        tree_ = nullptr;
    }


    KdTree::~KdTree() {
        delete get_tree(tree_);
    }


    void KdTree::begin() {
        delete get_tree(tree_);
        tree_ = nullptr;
    }


    void KdTree::end() {
        PointSet* pset = new PointSet(points_);
        KD_Tree* tree = new KD_Tree(pset);
        tree->buildIndex();
        tree_ = tree;
    }


    void KdTree::add_point_cloud(PointCloud* cloud) {
        points_ = &cloud->points();
    }


    int KdTree::find_closest_point(const vec3& p, float& squared_distance) const {
        std::size_t index;

        nanoflann::KNNResultSet<float> result_set(1);
        result_set.init(&index, &squared_distance);

        get_tree(tree_)->findNeighbors(result_set, p, nanoflann::SearchParams(10));
        return index;
    }


    int KdTree::find_closest_point(const vec3& p) const {
        float dist = 0;
        return find_closest_point(p, dist);
    }


    void KdTree::find_closest_K_points(
        const vec3& p, int k, std::vector<int>& neighbors, std::vector<float>& squared_distances
    )  const
    {
        std::vector<size_t> indices(k);
        std::vector<float>	sqr_distances(k);

        nanoflann::KNNResultSet<float> result_set(k);
        result_set.init(&indices[0], &sqr_distances[0]);
        get_tree(tree_)->findNeighbors(result_set, p, nanoflann::SearchParams(10));

        neighbors = std::vector<int>(indices.begin(), indices.end());
        squared_distances = sqr_distances;
    }


    void KdTree::find_closest_K_points(
        const vec3& p, int k, std::vector<int>& neighbors
    )  const
    {
        std::vector<float> squared_distances;
        return find_closest_K_points(p, k, neighbors, squared_distances);
    }


    void KdTree::find_points_in_radius(
        const vec3& p, float radius, std::vector<int>& neighbors, std::vector<float>& squared_distances
    )  const {
        std::vector<std::pair<std::size_t, float> >   matches;
        nanoflann::SearchParams params;
        params.sorted = false;
        const std::size_t num = get_tree(tree_)->radiusSearch(p, radius, matches, params);

        neighbors.resize(num);
        squared_distances.resize(num);
        for (std::size_t i = 0; i < num; ++i) {
            const std::pair<std::size_t, float>& e = matches[i];
            neighbors[i] = e.first;
            squared_distances[i] = e.second;
        }
    }


    void KdTree::find_points_in_radius(
        const vec3& p, float radius, std::vector<int>& neighbors
    )  const
    {
        std::vector<float> sqr_distances;
        return find_points_in_radius(p, radius, neighbors, sqr_distances);
    }


} // namespace easy3d
