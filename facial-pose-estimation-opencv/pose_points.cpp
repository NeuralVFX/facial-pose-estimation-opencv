#include "pose_points.h"
#include "utils.h"


pose_points::pose_points()
{
}


pose_points::pose_points(bool lock_eyes_nose)
{
	// Populate needed points of base pose into smaller array
	base_shape = cv::Mat(tri_len, 3, CV_32FC1);
	int count;
	count = 0;

	for (int id : triangulation_ids)
	{
		base_shape.at<float>(count, 0) = base_shape_arr[id][0];
		base_shape.at<float>(count, 1) = base_shape_arr[id][1];
		base_shape.at<float>(count, 2) = base_shape_arr[id][2];
		count++;
	}
	// Populate needed points of blendhshapes into smaller blendshape array
	for (int b = 0; b < 51; b++)
	{
		cv::Mat new_bs = cv::Mat(tri_len, 3, CV_32FC1);
		new_bs = 0;
		count = 0;

		for (int id : triangulation_ids)
		{
			// Option to skip over eyes and nose, sometimes more stable
			if (lock_eyes_nose)
			{
				if (id != 33 || id != 24 || id != 15 || id != 18)
				{
					new_bs.at<float>(count, 0) = blend_shapes_arr[b][id][0];
					new_bs.at<float>(count, 1) = blend_shapes_arr[b][id][1];
					new_bs.at<float>(count, 2) = blend_shapes_arr[b][id][2];
				}

			}
			else
			{
				new_bs.at<float>(count, 0) = blend_shapes_arr[b][id][0];
				new_bs.at<float>(count, 1) = blend_shapes_arr[b][id][1];
				new_bs.at<float>(count, 2) = blend_shapes_arr[b][id][2];
			}
			count++;
		}
		blend_shapes.push_back(new_bs);
	}
}


std::vector<cv::Point3d> pose_points::get_pose(cv::Mat expression)
{
	cv::Mat pose = cv::Mat(tri_len, 3, CV_32FC1);
	pose = 0;

	// Multiply blendshapes by values
	for (int i = 0; i < 51; i++)
	{
		float weight = expression.at<float>(0, i);
		weight = utils::clamp(weight, 0.0f, 1.0f);
		pose = pose + (weight*blend_shapes[i]);
	}
	// Add resulting blenshape to base pose
	pose = pose + base_shape;

	// Save output points to new array
	std::vector<cv::Point3d> out_points;
	out_points.clear();
	for (int i = 0; i < tri_len; i++)
	{
		cv::Point3d new_point((pose.at<float>(i, 0))*5000.f, (pose.at<float>(i, 1))*5000.f, (pose.at<float>(i, 2))*5000.f);
		out_points.push_back(new_point);
	}
	return out_points;
}


pose_points::~pose_points()
{
}
