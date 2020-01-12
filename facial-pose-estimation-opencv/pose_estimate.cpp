#pragma once

#include <iostream>
#include <stdio.h>
#include <cstdio>

#include "pose_estimate.h"
#include "utils.h"


Estimator::Estimator()
{
	// Load networks
	deep_expression = cv::dnn::readNet(expression_model);
	deep_expression.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL_FP16);
	box_detector = cv::dnn::readNetFromCaffe(caffe_config_file, caffe_weight_file);
	box_detector.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL_FP16);
	landmark_detector = dlib::shape_predictor();
	dlib::deserialize(landmarks_model) >> landmark_detector;

	// Set starting face box value
	face_rect = dlib::rectangle(dlib::point(0, 0), dlib::point(1, 1));

	// Set resolution
	frame_width = 1920;
	frame_height = 1080;
	run_count = 0;

	// Face detection res
	face_detect_res = 96;
	line_render_res = 96;
}

int Estimator::init(int& outCameraWidth, int& outCameraHeight, int detectRatio, int camId, float fovZoom, bool draw, bool lockEyesNose)
{
	// Set parameters from Unity
	lock_eyes_nose = lockEyesNose;
	fov_zoom = fovZoom;
	detect_ratio = detectRatio;
	draw_points = draw;

	// Pose object
	pose = pose_points(lock_eyes_nose);

	_capture.open(camId);
	if (!_capture.isOpened())
		return -2;

	_capture.set(cv::CAP_PROP_FRAME_WIDTH, frame_width);
	_capture.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height);

	frame_width = _capture.get(cv::CAP_PROP_FRAME_WIDTH);
	frame_height = _capture.get(cv::CAP_PROP_FRAME_HEIGHT);
	return 0;
}

void Estimator::close()
{
	_capture.release();
}

void Estimator::detect(TransformData& outFaces, ExpressionData* outExpression)
{
	_capture >> frame;
	if (frame.empty())
		return;

	bounding_box_detect();

	landmark_detect();

	landmark_to_blendshapes(outExpression);

	pnp_solve(outFaces);

	if (draw_points)
	{
		// Draw points and axis on face
		draw_solve();
	}

	run_count += 1;
	cv::waitKey(1);
}


void Estimator::get_raw_image_bytes(unsigned char* data, int width, int height)
{
	if (frame.empty())
		return;

	cv::Mat tex_mat(height, width, frame.type());
	cv::resize(frame, tex_mat, tex_mat.size(), cv::INTER_CUBIC);

	//Convert from RGB to ARGB 
	cv::Mat argb_img;
	cv::cvtColor(tex_mat, argb_img, cv::COLOR_RGB2BGRA);
	vector<cv::Mat> bgra;
	cv::split(argb_img, bgra);
	cv::swap(bgra[0], bgra[3]);
	cv::swap(bgra[1], bgra[2]);
	// Copy data back to pointer
	memcpy(data, argb_img.data, argb_img.total() * argb_img.elemSize());
}


void Estimator::draw_solve()
{
	// Build axis from nose
	vector<cv::Point3d> pose_axis_3d;
	vector<cv::Point2d> pose_axis_2d;
	pose_axis_3d.push_back(predicted_points_3d[0] + cv::Point3d(0, 0, 400.0));
	pose_axis_3d.push_back(predicted_points_3d[0] + cv::Point3d(400, 0, 0));
	pose_axis_3d.push_back(predicted_points_3d[0] + cv::Point3d(0, 400, 0));

	// Project points
	vector<cv::Point2d> predicted_face_2d;
	cv::projectPoints(predicted_points_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs, predicted_face_2d);
	cv::projectPoints(pose_axis_3d, rotation_vector, translation_vector, camera_matrix, dist_coeffs, pose_axis_2d);

	// Draw points
	for (int i = 0; i < 6; i++)
	{
		cv::circle(frame, cv::Point(landmark_points_2d[i].x, landmark_points_2d[i].y), 4, cv::Scalar(255, 0, 0), 3);
		cv::circle(frame, cv::Point(predicted_face_2d[i].x, predicted_face_2d[i].y), 4, cv::Scalar(0, 0, 255), 3);
	}

	// Draw axis lines
	cv::line(frame, predicted_face_2d[0], pose_axis_2d[0], cv::Scalar(255, 0, 0), 2);
	cv::line(frame, predicted_face_2d[0], pose_axis_2d[1], cv::Scalar(0, 255, 0), 2);
	cv::line(frame, predicted_face_2d[0], pose_axis_2d[2], cv::Scalar(0, 0, 255), 2);
}


void Estimator::pnp_solve(TransformData& outFaces)
{
	// Retrieve facial points with blenshapes applied
	predicted_points_3d = pose.get_pose(expression);

	// Prepair face points for perspective solve
	landmark_points_2d.clear();
	for (int id : triangulation_ids)
	{
		landmark_points_2d.push_back(cv::Point2d(face_landmarks.part(id).x() * detect_ratio, face_landmarks.part(id).y() * detect_ratio));
	}

	// Generate fake camera matrix
	double focal_length = frame.cols*fov_zoom;
	cv::Point2d center = cv::Point2d(frame.cols / 2, frame.rows / 2);
	camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
	dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type);

	// Output rotation and translation, defaulting to in front of the camera
	translation_vector.at< double>(0) = 0;
	translation_vector.at< double>(1) = 0;
	translation_vector.at< double>(2) = 3200;

	rotation_vector.at< double>(0) = -3.2f;
	rotation_vector.at< double>(1) = 0.0f;
	rotation_vector.at< double>(2) = 0.0f;
	cv::Mat rot_mat;

	// Solve for pose
	cv::solvePnP(predicted_points_3d, landmark_points_2d, camera_matrix, dist_coeffs, rotation_vector, translation_vector, true, cv::SOLVEPNP_ITERATIVE);

	// Convert rotation to Matrix
	cv::Rodrigues(rotation_vector, rot_mat);

	// Export transform
	outFaces = TransformData(translation_vector.at<double>(0), translation_vector.at<double>(1), translation_vector.at<double>(2),
		rot_mat.at<double>(2, 0), rot_mat.at<double>(2, 1), rot_mat.at<double>(2, 2),
		rot_mat.at<double>(1, 0), rot_mat.at<double>(1, 1), rot_mat.at<double>(1, 2));
}


void Estimator::landmark_to_blendshapes(ExpressionData* outExpression)
{
	// Construct line image for Expression Detection
	cv::Mat bs_mat = get_line_face(face_landmarks);
	cv::Mat bs_mat_flipped, bs_mat_32;
	cv::flip(bs_mat, bs_mat_flipped, 1);

	// Feed to network to get prediction
	bs_mat_flipped.convertTo(bs_mat_32, CV_32F);
	bs_mat_32 /= 127.5;
	bs_mat_32 -= cv::Scalar(1, 1, 1);

	cv::Mat expression_blob = cv::dnn::blobFromImage(bs_mat_32, 1, cv::Size(face_detect_res, face_detect_res), (0, 0, 0), false, false, CV_32F);
	deep_expression.setInput(expression_blob);
	expression = deep_expression.forward();

	// Clamp network outputs, set data into struct
	for (int i = 0; i < 51; i++)
	{
		float weight = expression.at<float>(0, i);
		weight = utils::clamp(weight, 0.0f, 1.0f);
		outExpression[i] = ExpressionData(weight);
	}
}


void Estimator::landmark_detect()
{
	// Run landmark detection
	cv::Mat half_frame(frame_height / detect_ratio, frame_width / detect_ratio, frame.type());
	cv::resize(frame, half_frame, half_frame.size(), cv::INTER_CUBIC);
	dlib::cv_image<dlib::bgr_pixel> dlib_image(half_frame);
	face_landmarks = landmark_detector(dlib_image, face_rect);
	// Store nose point
	prev_nose = cv::Point2f(face_landmarks.part(34).x(), face_landmarks.part(34).y());
}


void Estimator::bounding_box_detect()
{
	// Convert frame to blob, and drop into Face Box Detector Netowrk
	cv::Mat blob, out;
	blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), (104, 117, 123), false, false);

	box_detector.setInput(blob);
	cv::Mat detection = box_detector.forward();
	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	// Check results and only take the most confident prediction
	float largest_conf = 0;
	for (int i = 0; i < detectionMat.rows; i++)
	{
		float confidence = detectionMat.at<float>(i, 2);

		if (confidence > .5)
		{
			// Get dimensions
			int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * (frame_width / detect_ratio));
			int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * (frame_height / detect_ratio));
			int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * (frame_width / detect_ratio));
			int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * (frame_height / detect_ratio));

			// Generate square dimensions
			face_width = max(x2 - x1, y2 - y1) / 2.8;
			center_x = ((x2 + x1) / 2);
			center_y = ((y2 + y1) / 2);

			if (run_count > 0)
			{
				// Average the center of the box with the Nose (Works better for landmark detection)
				center_x = int((center_x + prev_nose.x) / 2);
				center_y = int((center_y + prev_nose.y) / 2);
			}

			// Apply square dimensions
			dlib::point point_a(center_x - face_width, center_y - face_width);
			dlib::point point_b(center_x + face_width, center_y + face_width);
			dlib::rectangle new_face(point_a, point_b);

			if (confidence > largest_conf)
			{
				largest_conf = confidence;
				face_rect = new_face;
			}
		}
	}
}


cv::Mat Estimator::get_line_face(dlib::full_object_detection face_landmark)
{
	// Draw line image from Face Landmarks
	int point_size = point_ids.size();
	cv::Mat x_points(point_size, 1, CV_32FC1);
	cv::Mat y_points(point_size, 1, CV_32FC1);
	int count = 0;
	for (int id : point_ids)
	{
		x_points.at<float>(count, 0) = face_landmark.part(id).x();
		y_points.at<float>(count, 0) = face_landmark.part(id).y();
		count++;
	}
	cv::Mat bs_mat = cv::Mat::zeros(line_render_res, line_render_res, CV_8UC3);
	bs_mat = 0;

	// Measure value range, and then center values
	double min_val, max_val;
	cv::minMaxLoc(x_points, &min_val, &max_val);
	double width = max_val - min_val;
	double mean_x = (max_val + min_val) / 2;
	cv::minMaxLoc(y_points, &min_val, &max_val);
	double height = max_val - min_val;
	double mean_y = (max_val + min_val) / 2;

	x_points = x_points - mean_x;
	y_points = y_points - mean_y;
	x_points = (x_points / std::max(width, height))*(line_render_res * .9);
	y_points = (y_points / std::max(width, height))*(line_render_res * .9);
	x_points = x_points + (line_render_res / 2);
	y_points = y_points + (line_render_res / 2);

	vector<cv::Scalar>temp_colors(draw_colors);

	// Chin
	utils::build_line(&bs_mat, x_points, y_points, vector<int>{0, 1, 2, 3, 4}, temp_colors.back());
	temp_colors.pop_back();
	// Right Eye Broww
	utils::build_line(&bs_mat, x_points, y_points, vector<int>{5, 6, 7, 8, 9}, temp_colors.back());
	temp_colors.pop_back();
	// Left Eye Brow
	utils::build_line(&bs_mat, x_points, y_points, vector<int>{10, 11, 12, 13, 14}, temp_colors.back());
	temp_colors.pop_back();
	// Eyes
	utils::build_line(&bs_mat, x_points, y_points, vector<int>{24, 25, 26, 27}, temp_colors.back());
	temp_colors.pop_back();

	utils::build_line(&bs_mat, x_points, y_points, vector<int>{27, 28, 29, 24}, temp_colors.back());
	temp_colors.pop_back();

	utils::build_line(&bs_mat, x_points, y_points, vector<int>{30, 31, 32, 33}, temp_colors.back());
	temp_colors.pop_back();

	utils::build_line(&bs_mat, x_points, y_points, vector<int>{33, 34, 35, 30}, temp_colors.back());
	temp_colors.pop_back();
	// Outer Mouth
	utils::build_line(&bs_mat, x_points, y_points, vector<int>{36, 37, 38, 39}, temp_colors.back());
	temp_colors.pop_back();

	utils::build_line(&bs_mat, x_points, y_points, vector<int>{39, 40, 41, 42}, temp_colors.back());
	temp_colors.pop_back();

	utils::build_line(&bs_mat, x_points, y_points, vector<int>{42, 43, 44, 45}, temp_colors.back());
	temp_colors.pop_back();

	utils::build_line(&bs_mat, x_points, y_points, vector<int>{45, 46, 47, 36}, temp_colors.back());
	temp_colors.pop_back();
	// Inner Mouth
	utils::build_line(&bs_mat, x_points, y_points, vector<int>{48, 49, 50}, temp_colors.back());
	temp_colors.pop_back();

	utils::build_line(&bs_mat, x_points, y_points, vector<int>{50, 51, 52}, temp_colors.back());
	temp_colors.pop_back();

	utils::build_line(&bs_mat, x_points, y_points, vector<int>{52, 53, 54}, temp_colors.back());
	temp_colors.pop_back();

	utils::build_line(&bs_mat, x_points, y_points, vector<int>{54, 55, 48}, temp_colors.back());
	temp_colors.pop_back();
	// Nose
	utils::build_line(&bs_mat, x_points, y_points, vector<int>{15, 16, 17, 18}, temp_colors.back());
	temp_colors.pop_back();

	utils::build_line(&bs_mat, x_points, y_points, vector<int>{18, 19}, temp_colors.back());
	temp_colors.pop_back();

	utils::build_line(&bs_mat, x_points, y_points, vector<int>{19, 20, 21}, temp_colors.back());
	temp_colors.pop_back();

	utils::build_line(&bs_mat, x_points, y_points, vector<int>{21, 22, 23}, temp_colors.back());
	temp_colors.pop_back();

	utils::build_line(&bs_mat, x_points, y_points, vector<int>{18, 23}, temp_colors.back());
	temp_colors.pop_back();

	return bs_mat;

}

