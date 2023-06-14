#include "Block.h"
#include "Maze.h"
#include "Agent.h"
#include "utils.h"
#include <cstdlib>
#include <unistd.h> // for usleep
#include <math.h>
#include <bits/stdc++.h>
#include <vector>
#include <string>
#include <list>
#include <iostream>
#include <stdio.h>

#include <cstdio>


using namespace std;

Block::Block(float x, float y, float w, float h ):x(x),y(y),w(w),h(h){
		bl={x,y};
		ul={x,y+h};
		br={x+w,y};
		ur={x+w,y+h};
	}


	
tuple<bool,vector<float>,int> Block::intersect(vector<float> start, vector<float> end, float r){
	int wall=0;
	vector<float> i={FLT_MAX,FLT_MAX};
	vector<tuple<vector<float>,int>> intersections;
	// left
	tuple<bool,vector<float>> ileft=LineIntersection(start, end, bl, ul);

	if (get<0>(ileft) && in(start, end, get<1>(ileft), 1) && y<=get<1>(ileft)[1] && get<1>(ileft)[1]<=y+h)
		{intersections.push_back(make_tuple(get<1>(ileft),1));}
	// right
	tuple<bool,vector<float>> iright=LineIntersection(start, end, br, ur);
	if (get<0>(iright) && in(start, end, get<1>(iright), 2) && y<=get<1>(iright)[1] && get<1>(iright)[1]<=y+h)
		{intersections.push_back(make_tuple(get<1>(iright),2));}
	// bottom
	tuple<bool,vector<float>> ibottom=LineIntersection(start, end, bl, br);
	if (get<0>(ibottom) && in(start, end, get<1>(ibottom), 3) && x<=get<1>(ibottom)[0] && get<1>(ibottom)[0]<=x+w)
		{intersections.push_back(make_tuple(get<1>(ibottom),3));}
	// top
	tuple<bool,vector<float>> itop=LineIntersection(start, end, ul, ur);
	if (get<0>(itop) && in(start, end, get<1>(itop), 4)&& x<=get<1>(itop)[0] && get<1>(itop)[0]<=x+w)
		{intersections.push_back(make_tuple(get<1>(itop),4));}
		
	// const char* c = to_string(intersections.size()).c_str();
	// printf(c);
	// printf("\n");
	if (intersections.size()>0)
		{i=get<0>(intersections[0]);
		wall=get<1>(intersections[0]);
		for (tuple<vector<float>,int> point : intersections){
			if (dist(start,get<0>(point))<dist(start,i))
				{i=get<0>(point);
				wall=get<1>(point);}}
		return make_tuple(true,i,wall);}
	else
		{return make_tuple(false,i,wall);}
	

}