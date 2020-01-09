#pragma once

#include <iostream>
#include <stdio.h>
#include <cstdio>

#include "pose_estimate.h"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/dnn/shape_utils.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"

class utils
{
public:
	utils();

	static void build_line(cv::Mat *image, cv::Mat x_points, cv::Mat y_points, vector<int> id_list, cv::Scalar color);

	static float clamp(float value, float low, float high);

	~utils();
};

