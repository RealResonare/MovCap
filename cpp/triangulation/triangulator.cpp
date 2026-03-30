#include "triangulator.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace mocap {

void Triangulator::addCamera(int camera_id, const CameraParams& params) {
    cameras_[camera_id] = params;
}

Eigen::Vector3d Triangulator::triangulatePair(
    const Eigen::Vector2d& p1,
    const Eigen::Vector2d& p2,
    const CameraParams& cam1,
    const CameraParams& cam2
) const {
    Eigen::Matrix4d A;
    A.row(0) = p1.x() * cam1.P.row(2) - cam1.P.row(0);
    A.row(1) = p1.y() * cam1.P.row(2) - cam1.P.row(1);
    A.row(2) = p2.x() * cam2.P.row(2) - cam2.P.row(0);
    A.row(3) = p2.y() * cam2.P.row(2) - cam2.P.row(1);

    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d X = svd.matrixV().col(3);

    if (std::abs(X(3)) < 1e-10) {
        return Eigen::Vector3d::Zero();
    }

    return X.head<3>() / X(3);
}

Eigen::Vector3d Triangulator::triangulateDLT(
    const std::vector<Eigen::Vector2d>& points_2d,
    const std::vector<CameraParams>& params
) const {
    int n = static_cast<int>(points_2d.size());
    Eigen::MatrixXd A(2 * n, 4);

    for (int i = 0; i < n; ++i) {
        double x = points_2d[i].x();
        double y = points_2d[i].y();
        A.row(2 * i) = x * params[i].P.row(2) - params[i].P.row(0);
        A.row(2 * i + 1) = y * params[i].P.row(2) - params[i].P.row(1);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::Vector4d X = svd.matrixV().col(3);

    if (std::abs(X(3)) < 1e-10) {
        return Eigen::Vector3d::Zero();
    }

    return X.head<3>() / X(3);
}

double Triangulator::reprojectError(
    const Eigen::Vector3d& point_3d,
    const Eigen::Vector2d& point_2d,
    const CameraParams& params
) const {
    Eigen::Vector4d X;
    X.head<3>() = point_3d;
    X(3) = 1.0;

    Eigen::Vector3d proj = params.P * X;
    Eigen::Vector2d proj_2d(proj(0) / proj(2), proj(1) / proj(2));

    return (proj_2d - point_2d).norm();
}

TriangulatedPoint Triangulator::triangulateJoint(
    const std::vector<Eigen::Vector2d>& points_2d,
    const std::vector<int>& camera_ids
) const {
    std::vector<CameraParams> cam_params;
    for (int id : camera_ids) {
        auto it = cameras_.find(id);
        if (it != cameras_.end()) {
            cam_params.push_back(it->second);
        }
    }

    Eigen::Vector3d point;
    if (cam_params.size() == 2) {
        point = triangulatePair(points_2d[0], points_2d[1],
                                cam_params[0], cam_params[1]);
    } else {
        point = triangulateDLT(points_2d, cam_params);
    }

    double total_err = 0.0;
    int count = 0;
    for (size_t i = 0; i < points_2d.size() && i < cam_params.size(); ++i) {
        total_err += reprojectError(point, points_2d[i], cam_params[i]);
        ++count;
    }

    return TriangulatedPoint{
        point,
        count > 0 ? total_err / count : std::numeric_limits<double>::infinity(),
        static_cast<int>(cam_params.size()),
    };
}

std::vector<TriangulatedPoint> Triangulator::triangulatePose(
    const std::vector<std::vector<Eigen::Vector2d>>& joint_points_2d,
    const std::vector<int>& camera_ids
) const {
    std::vector<TriangulatedPoint> results;
    results.reserve(joint_points_2d.size());

    for (const auto& pts : joint_points_2d) {
        results.push_back(triangulateJoint(pts, camera_ids));
    }

    return results;
}

}  // namespace mocap
