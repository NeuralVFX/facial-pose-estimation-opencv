#include "utils.h"



utils::utils()
{
}


void utils::build_line(cv::Mat *image, cv::Mat x_points, cv::Mat y_points, vector<int> id_list, cv::Scalar color)
{
	vector<cv::Point> new_point_list;
	cv::Mat line_image(image->rows, image->cols, CV_8UC3);
	line_image = 0;

	for (int id : id_list)
	{
		new_point_list.push_back(cv::Point(x_points.at<float>(id, 0), y_points.at<float>(id, 0)));
	}
	cv::polylines(line_image, new_point_list, false, color, 1, cv::LINE_AA, 0);
	cv::max(line_image, *image, *image);
}

float utils::clamp(float start_value, float low, float high)
{
	float value = start_value;

	if (value > high)
	{
		value = high;
	}
	else if (value < low)
	{
		value = low;
	}
	return value;
}


utils::~utils()
{
}
