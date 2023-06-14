
#ifndef __MAZE_H__
#define __MAZE_H__
#include "Agent.h"
#include "Block.h"
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


// namespace py = pybind11;

class Agent;// necessary incomplete declaration
class Block;

class Maze
{   
	public:
		// VARIABLES
		int width=100;
		int height=100;
		float xgoal=90.0;
		float ygoal=90.0;
		float xinit=5.0;
		float yinit=5.0;
		int time_horizon=1000;
		int time=0;
		float dt=0.1;
		float vmax=500.0;
		float vhit=100.0;
		float treshold=1;
		float life_penalty=-1;
	
		// Block
		vector<Block> block_list;
		// Agent
		Agent *agent;
		// Vibes
		bool figure=false;


		// FUNCTIONS
		Maze(float xinit, float yinit, float xgoal, float ygoal, int time_horizon , int n_b);

		vector<float> reset();
		
		tuple<vector<float>,float,bool,string> step(const vector<float> &action);
		
		void add_block(float x, float y, float w, float h);

		void render();
		
};

#endif
