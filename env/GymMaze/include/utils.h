#ifndef __UTILS_H__
#define __UTILS_H__

#include <cstdlib>
#include <unistd.h> // for usleep
#include <math.h>
#include <bits/stdc++.h>
#include <vector>
#include <string>
#include <list>
#include <iostream>



using namespace std;

vector<float> linspace(float start, float stop, int nb = 10);

tuple<bool,vector<float>> LineIntersection(vector<float> A, vector<float> B, vector<float> C, vector<float> D);


bool in(vector<float> start, vector<float> end, vector<float> i, int wall);
	

float dist(vector<float> start, vector<float> point);
	
#endif