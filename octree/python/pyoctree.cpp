#include <string>

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <points.h>
#include <octree_samples.h>
#include <marching_cube.h>

// qq add for octree
#include <octree.h>
#include <octree_parser.h>
#include <stdint.h>



namespace py = pybind11;

int add(int i, int j) {
  return i + j;
}

py::array_t<double> make_array(const py::ssize_t size) {
  // No pointer is passed, so NumPy will allocate the buffer
  return py::array_t<double>(size);
}

PYBIND11_MODULE(pyoctree, m) {
  // examples
  m.def("add", &add);
  m.def("subtract", [](int i, int j) { return i - j; });

  //m.def("make_array", &make_array,
  //  py::return_value_policy::move);
  // Return policy can be left default, i.e. return_value_policy::automatic

  // pyoctree interface
  m.def("get_one_octree", [](const char *name) {
    size_t size = 0;
    const char* str = (const char*)octree::get_one_octree(name, &size);
    return py::bytes(std::string(str, size));
  });


  using vectorf = vector<float>;
  using vectorfc = const vector<float>;
  auto Points_set_points = (bool(Points::*)(vectorfc&, vectorfc&, vectorfc&,
              vectorfc&))&Points::set_points;
  auto Mcube_compute = (void(MarchingCube::*)(vectorfc&, float, vectorfc&,
              int))&MarchingCube::compute;

  // points interface
  py::class_<Points>(m, "Points")
  .def(py::init<>())
  .def("read_points", &Points::read_points)
  .def("write_points", &Points::write_points)
  .def("write_ply", &Points::write_ply)
  .def("normalize", &Points::normalize)
  .def("orient_normal", &Points::orient_normal)
  .def("set_points", Points_set_points)
//  .def("set_points_buffer", Points_set_points_buffer)
  .def("pts_num", [](const Points& pts) {
    return pts.info().pt_num();
  })
  // todo: fix the functions points(), normals(), labels(),
  // It is inefficient, since there is a memory-copy here
  .def("points", [](const Points & pts) {
    const float* ptr = pts.points();
    int num = pts.info().pt_num();
    return vectorf(ptr, ptr + num * 3);
  })
  .def("normals", [](const Points & pts) {
    const float* ptr = pts.normal();
    int num = pts.info().pt_num();
    return vectorf(ptr, ptr + num * 3);
  })
  .def("features", [](const Points & pts) {
    const float* ptr = pts.feature();
    int num = pts.info().pt_num();
    const int ch = pts.info().channel(PointsInfo::kFeature);
    return vectorf(ptr, ptr + num * ch);
  })
  .def("labels", [](const Points & pts) {
    const float* ptr = pts.label();
    int num = pts.info().pt_num();
    return vectorf(ptr, ptr + num);
  })
  .def("buffer", [](const Points & pts) {
    const char* ptr = pts.data();
    return py::bytes(std::string(ptr, pts.info().sizeof_points()));
  });

  // marching cube
  py::class_<MarchingCube>(m, "MCube")
  .def(py::init<>())
  .def("compute", Mcube_compute)
  .def("get_vtx", [](const MarchingCube & mcube) {
    return mcube.vtx_;
  })
  .def("get_face", [](const MarchingCube & mcube) {
    return mcube.face_;
  });

  // qq: try octree interface
  using vectoru = vector<uint32_t>;
  using vectori = vector<int>;
  py::class_<Octree>(m, "Octree")
  .def(py::init<>())
  .def("read_octree", &Octree::read_octree)
  .def("write_octree", &Octree::write_octree)
//  .def("info", &Octree::info)
  .def("depth", [](const Octree& octree) {
    return octree.info().depth();
  })
  .def("num_nodes", [](const Octree& octree, int depth) {
//    int depth = octree.info().depth();
    int num = octree.info().node_num(depth);
    return num;
  })
  .def("num_nodes_total", [](const Octree& octree) {
    int num = octree.info().total_nnum();
    return num;
  })
  .def("keys", [](const Octree& octree, int depth) {
//    int depth = octree.info().depth();
    const uintk* ptr = octree.key_cpu(depth);
    int num = octree.info().node_num(depth);
    return vectoru(ptr, ptr + num);
  })
  .def("features", [](const Octree& octree, int depth) {
//    int depth = octree.info().depth();
    const float* ptr = octree.feature_cpu(depth);
    int num = octree.info().node_num(depth);
    const int ch = octree.info().channel(OctreeInfo::kFeature);
    return vectorf(ptr, ptr + num * ch);
  })
  .def("info_size", [](const Octree& octree) {
    int info_size = octree.info().sizeof_octinfo();
    return info_size;
  })
  .def("children", [](const Octree& octree, int depth) {
//    int depth = octree.info().depth();
    const int* ptr = octree.children_cpu(depth);
    int num = octree.info().node_num(depth);
    return vectori(ptr, ptr + num);
  })
  .def("num_channel", [](const Octree& octree) {
    const int ch = octree.info().channel(OctreeInfo::kFeature);
    return ch;
  });
}
