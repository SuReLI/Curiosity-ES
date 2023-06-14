#include <cstdlib>
#include <unistd.h> // for usleep
#include <math.h>
#include <bits/stdc++.h>
#include <vector>
#include <string>
#include <list>
#include <iostream>


#define p vector<float>
using namespace std;

tuple<bool,vector<float>> LineIntersection(p A, p B, p C, p D)
{	// Line AB represented as a1x + b1y = c1
	float a1 = B[1] - A[1];
	float b1 = A[0] - B[0];
	float c1 = a1*(A[0]) + b1*(A[1]);
	// Line CD represented as a2x + b2y = c2
	float a2 = D[1] - C[1];
	float b2 = C[0] - D[0];
	float c2 = a2*(C[0])+ b2*(C[1]);
	float determinant = a1*b2 - a2*b1;
	if (determinant == 0)
	{ 	// return False, <FLT_MAX,FLT_MAX>
		p point{FLT_MAX,FLT_MAX};
		return make_tuple(false, point);}
	else
	{	float x = (b2*c1 - b1*c2)/determinant;
		float y = (a1*c2 - a2*c1)/determinant;
		p point{x,y};
		return make_tuple(true, point);}
}

bool in(p start, p end, p i, int wall){
	bool bin=false;
	if (wall == 1 || wall ==2)
	{	// X
		if (start[0]<=end[0])
			{if (start[0]<=i[0] && i[0]<=end[0])
				{bin=true;}}
			
		else
			{if (end[0]<=i[0] && i[0]<=start[0])
				{bin=true;}}
	}
	else
	{	// Y
		if (start[1]<=end[1])
			{if (start[1]<=i[1] && i[1]<=end[1])
				{bin=true;}}
		else
			{if (end[1]<=i[1] && i[1]<=start[1])
				{bin=true;}}}
	
	return bin;
}

float dist(p start, p point){
	return sqrt(pow(point[0]-start[0],2)+pow(point[1]-start[1],2));
}

vector<float> linspace(float start, float stop, int nb) {
    vector<float> values;
	for (int i = 0; i < nb; i++)
	{values.push_back((float)i*(stop-start)/nb+start);}
	
	
    
    return values;
}