#ifndef __AGENT_H__
#define __AGENT_H__
#include "Maze.h"
#include "Block.h"
#include "utils.h"
#include <cstdlib>
#include <unistd.h> // for usleep
#include <math.h>
#include <bits/stdc++.h>
#include <vector>
#include <string>
#include <list>
#include <iostream>


// #define p vector<float>
using namespace std;

class Maze;// necessary incomplete declaration
class Block;

class Agent
{   
	public:

		// VARIABLES
		// Eigen::Vector2d position;
		float x;
		float y; 
		float xdot=0.0;
		float ydot=0.0;
		float v = 0;
		float theta=0;
		float dt;
		float r=1.0;
		// Vmax
		float vmax;
		float vhit;

		// lidar 
		int n_beams;
		vector<float> beams;
		float lidar_range;
		float max_beam;
		// Maze
		Maze *maze;

		// FUNCTIONS
		Agent(Maze *maze, float xinit, float yinit, float dt, float vmax, float vhit, int n_b);
		bool move(vector<float>action);
		vector<float> lidar_observation();

};

#endif