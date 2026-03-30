#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "triangulator.h"

namespace py = pybind11;

PYBIND11_MODULE(mocap_core, m) {
    m.doc() = "MovCap C++ acceleration module";

    py::class_<mocap::CameraParams>(m, "CameraParams")
        .def(py::init<>())
        .def_readwrite("K", &mocap::CameraParams::K)
        .def_readwrite("P", &mocap::CameraParams::P)
        .def_readwrite("dist", &mocap::CameraParams::dist);

    py::class_<mocap::TriangulatedPoint>(m, "TriangulatedPoint")
        .def(py::init<>())
        .def_readwrite("point", &mocap::TriangulatedPoint::point)
        .def_readwrite("reprojection_error", &mocap::TriangulatedPoint::reprojection_error)
        .def_readwrite("num_views", &mocap::TriangulatedPoint::num_views);

    py::class_<mocap::Triangulator>(m, "Triangulator")
        .def(py::init<>())
        .def("add_camera", &mocap::Triangulator::addCamera)
        .def("triangulate_joint", &mocap::Triangulator::triangulateJoint)
        .def("triangulate_pose", &mocap::Triangulator::triangulatePose)
        .def("reproject_error", &mocap::Triangulator::reprojectError);
}
