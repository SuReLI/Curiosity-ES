#ifndef __BLOCK_H__
#define __BLOCK_H__
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



class Block
{   
	public:
		// VARIABLES
		float x;
		float y; 
		float w;
		float h; 
		vector<float> bl;
		vector<float> ul;
		vector<float> br;
		vector<float> ur;
		// FUNCTIONS
		Block(float x, float y, float w, float h );

		tuple<bool,vector<float>,int> intersect(vector<float> start, vector<float> end, float r);
};

#endif