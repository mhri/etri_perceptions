#pragma once

#include <string>
#include <vector>
using namespace std;

typedef struct
{
	int x;
	int y;

} Position;


class CMindStateRecog
{
public:
	CMindStateRecog(void);
	~CMindStateRecog(void);


	bool LoadMindStateModel(string modelFilePath, int nBaseFrameSize=5);
	string MindRecogRes(vector<vector<Position> > &dataVec);

};
