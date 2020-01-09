#pragma once

#include <iostream>
#include <stdio.h>
#include <cstdio>

#include "pose_estimate.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"


using namespace std;


class utils
{
public:
	utils();

	static void build_line(cv::Mat *image, cv::Mat x_points, cv::Mat y_points, vector<int> id_list, cv::Scalar color);

	static float clamp(float value, float low, float high);

	~utils();
};
