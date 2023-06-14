#include "Maze.h"
#include "Agent.h"
#include "Block.h"
#include "utils.h"
#include <cstdlib>
#include <unistd.h> // for usleep
#include <math.h>
#include "vibes.h"
#include <bits/stdc++.h>
#include <vector>
#include <string>
#include <list>
#include <iostream>
// #include <eigen3/Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
// This pair is used to store the X and Y
// coordinates of a point respectively
// using Eigen::Matrix;
// using namesapce Eigen;
#define p vector<float>

using namespace std;
namespace py = pybind11;



PYBIND11_MODULE(Game, m) {
	py::class_<Maze>(m, "Maze")
		.def(py::init<float, float,float, float, int, int>(), py::arg("xinit") = 5.0, py::arg("yinit") = 5.0, py::arg("xgoal") = 90.0, py::arg("ygoal") = 90.0, py::arg("time_horizon")=3000, py::arg("n_b")=5)
		.def("Cstep", &Maze::step)
		.def("Creset", &Maze::reset)
		.def("Crender",&Maze::render)
		.def("add_block",&Maze::add_block)
		.def_readwrite("agent", &Maze::agent)
		.def_readwrite("block_list", &Maze::block_list)
		.def_readwrite("xinit", &Maze::xinit)
		.def_readwrite("yinit", &Maze::yinit)
		.def_readwrite("xgoal", &Maze::xgoal)
		.def_readwrite("ygoal", &Maze::ygoal)
		.def_readwrite("width", &Maze::width)
		.def_readwrite("height", &Maze::height)
		.def_readwrite("vmax", &Maze::vmax)
		.def_readwrite("time_horizon", &Maze::time_horizon)
		.def_readwrite("time", &Maze::time)
		.def_readwrite("life_penalty", &Maze::life_penalty)
		.def_readwrite("treshold", &Maze::treshold);
	py::class_<Agent>(m, "Agent")
		.def(py::init<Maze *, float, float, float, float,float, int>())
		.def("move", &Agent::move)
		.def("lidar_observation", &Agent::lidar_observation)
		.def_readwrite("n_beams", &Agent::n_beams)
		.def_readwrite("beams", &Agent::beams)
		.def_readwrite("lidar_range", &Agent::lidar_range)
		.def_readwrite("r", &Agent::r)
		.def_readwrite("x", &Agent::x)
		.def_readwrite("y", &Agent::y)
		.def_readwrite("xdot", &Agent::xdot)
		.def_readwrite("ydot", &Agent::ydot);
	
	py::class_<Block>(m, "Block")
		.def(py::init<float, float, float, float>())
		.def_readwrite("x", &Block::x)
		.def_readwrite("y", &Block::y)
		.def_readwrite("w", &Block::w)
		.def_readwrite("h", &Block::h);

	

}
