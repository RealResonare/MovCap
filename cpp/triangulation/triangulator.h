#pragma once

#include <vector>
#include <Eigen/Dense>

namespace mocap {

struct CameraParams {
    Eigen::Matrix3d K;           // Intrinsic matrix
    Eigen::Matrix<double, 3, 4> P; // Projection matrix
    Eigen::VectorXd dist;        // Distortion coefficients
};

struct TriangulatedPoint {
    Eigen::Vector3d point;
    double reprojection_error;
    int num_views;
};

class Triangulator {
public:
    Triangulator() = default;

    void addCamera(int camera_id, const CameraParams& params);

    TriangulatedPoint triangulateJoint(
        const std::vector<Eigen::Vector2d>& points_2d,
        const std::vector<int>& camera_ids
    ) const;

    std::vector<TriangulatedPoint> triangulatePose(
        const std::vector<std::vector<Eigen::Vector2d>>& joint_points_2d,
        const std::vector<int>& camera_ids
    ) const;

    double reprojectError(
        const Eigen::Vector3d& point_3d,
        const Eigen::Vector2d& point_2d,
        const CameraParams& params
    ) const;

private:
    std::map<int, CameraParams> cameras_;

    Eigen::Vector3d triangulateDLT(
        const std::vector<Eigen::Vector2d>& points_2d,
        const std::vector<CameraParams>& params
    ) const;

    Eigen::Vector3d triangulatePair(
        const Eigen::Vector2d& p1,
        const Eigen::Vector2d& p2,
        const CameraParams& cam1,
        const CameraParams& cam2
    ) const;
};

}  // namespace mocap
