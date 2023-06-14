#include "Maze.h"
#include "Agent.h"
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
#include "vibes.h"


using namespace std;

Maze::Maze(float xinit, float yinit, float xgoal, float ygoal, int time_horizon , int n_b): xinit(xinit), yinit(yinit), xgoal(xgoal), ygoal(ygoal), time_horizon(time_horizon){
	agent=new Agent(this,xinit,yinit,dt,vmax, vhit, n_b);
	}
vector<float> Maze::reset(){
	// init agent 
	agent->x=xinit;
	agent->y=yinit;
	// state
	vector<float> state;
	int hit=0;
		// position
	state.push_back(agent->x/this->width);
	state.push_back(agent->y/this->height);
	state.push_back(2*sin(agent->x/this->width*2*M_PI/3)*(abs(agent->x-this->xgoal)));
	state.push_back(2*sin(agent->y/this->height*2*M_PI/3)*(abs(agent->y-this->ygoal)));
	state.push_back(agent->xdot/this->vmax*10);
	state.push_back(agent->ydot/this->vmax*10);
	// lidar
	vector<float> lidar= agent->lidar_observation();
	for(float beam : lidar){state.push_back(beam/agent->lidar_range);}
	state.push_back(hit);
	// time_horizon
	time=0;
	// lidar
	// TODO
	return state;

}

tuple<vector<float>,float,bool,string> Maze::step(const vector<float> &action){
	// update time
	time+=1;
	// move agent
	bool hit = agent->move(action);

	// done
	bool done=false;
	// done=hit;
	if (time>=time_horizon)
		{done=true;}

	// state
	vector<float> state;
	// position
	state.push_back(agent->x/this->width);
	state.push_back(agent->y/this->height);
	state.push_back(2*sin(agent->x/this->width*2*M_PI/3)*(abs(agent->x-this->xgoal)));
	state.push_back(2*sin(agent->y/this->height*2*M_PI/3)*(abs(agent->y-this->ygoal)));
	state.push_back(agent->xdot/this->vmax*2);
	state.push_back(agent->ydot/this->vmax*2);
	// state.push_back(hit*2);
	// lidar
	vector<float> lidar= agent->lidar_observation();
	for(float beam : lidar){state.push_back(beam/agent->lidar_range);}
	state.push_back(hit);
	// reward
	float distance=sqrt(pow(xgoal-agent->x,2)+pow(ygoal-agent->y,2));
	// float reward=this->life_penalty;
	float reward=0.0;
	// if (hit){
	// 	reward-=1.0;
	// }
	if (distance<this->treshold)
		{reward+=1;
		done=true;}
	// if (hit){
	// 	done=true;
	// }
	
	


	// info
	string info="nothing";

	tuple<vector<float>,float,bool,string> output(state,reward,done,info);
	return output;
}

void Maze::add_block(float x, float y, float w, float h){
	block_list.push_back(Block(x,y,w,h));
}

void Maze::render(){
	if (!figure)
	{	vibes::beginDrawing();
		vibes::newFigure("Maze");
		vibes::setFigureProperties("Maze", vibesParams("x",200, "y", 200, "width", 400, "height", 400));
		vibes::axisLimits(0., width, 0., height);
		figure=true;}
		
	vibes::clearFigure("Maze");
	// Agent
	vibes::drawCircle(agent->x, agent->y, agent->r, "red[blue]", vibesParams("figure", "Maze"));
	// goal
	vibes::drawCircle(xgoal, ygoal, agent->r, "red[green]", vibesParams("figure", "Maze"));
	// Blocks
	for (Block block : block_list)
	{
		vibes::drawBox(block.x,block.x+block.w,block.y,block.y+block.h, "black[black]", vibesParams("figure", "Maze"));
	}
	vector<float> lidar= agent->lidar_observation();
	for (int i = 0; i < agent->n_beams; i++)
	{	vector<double> x={agent->x,agent->x+lidar[i]*cos(agent->beams[i])};
		vector<double> y={agent->y,agent->y+lidar[i]*sin(agent->beams[i])};
		vibes::drawLine(x,y,"red[red]", vibesParams("figure", "Maze"));
	}
	
	usleep(dt * 500000.); 
	
}