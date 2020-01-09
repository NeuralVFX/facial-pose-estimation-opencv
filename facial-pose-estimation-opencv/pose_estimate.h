#pragma once
#include <iostream>
#include <stdio.h>
#include <cstdio>


#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/dnn/shape_utils.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "dlib/image_processing.h"
#include "dlib/opencv.h"
#include "dlib/image_processing/frontal_face_detector.h"

#include "pose_points.h"

using namespace std;


// Struct to pass data from DLL
struct TransformData
{
	TransformData(float tx, float ty, float tz, float rfx, float rfy, float rfz, float rux, float ruy, float ruz) :
		tX(tx), tY(ty), tZ(tz), rfX(rfx), rfY(rfy), rfZ(rfz), ruX(rux), ruY(ruy), ruZ(ruz) {}
	float tX, tY, tZ;
	float rfX, rfY, rfZ;
	float ruX, ruY, ruZ;
};

struct ExpressionData
{
	ExpressionData(float blendweight) :
		blend_weight(blendweight) {}
	float blend_weight;
};


class Estimator
{
public:

	cv::VideoCapture _capture;

	// Constant strings
	const string caffe_config_file = "./deploy.prototxt";
	const string caffe_weight_file = "./res10_300x300_ssd_iter_140000_fp16.caffemodel";
	const string landmarks_model = "./shape_predictor_68_face_landmarks.dat";
	const string expression_model = "./opt_model.onnx";

	// Face box data
	int face_width;
	int center_x;
	int center_y;

	// Tick counter
	int run_count;

	// Capture Dimensions
	int frame_width;
	int frame_height;
	int scale_ratio;

	// Face detection res and line render res
	int face_detect_res;
	int line_render_res;

	// Networks
	cv::dnn::Net box_detector;
	dlib::shape_predictor landmark_detector;
	cv::dnn::Net deep_expression;

	// Camera Zoom
	float fov_zoom;

	// Draw Axis on face
	bool draw_points;

	// Storage for reusable variables
	cv::Point2f prev_nose;
	vector< vector<cv::Point2f> > landmarks;
	dlib::rectangle face_rect;
	cv::Mat frame;

	// Storage for constant data arrays
	const vector<int> point_ids = { 6,7,8,9,10,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,
		33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67 };

	const vector<int> triangulation_ids = { 30,8, 36, 45, 48, 54 };

	const vector<cv::Scalar> draw_colors = { cv::Scalar(0,0,0),cv::Scalar(0,0,128),cv::Scalar(128,0,0),cv::Scalar(128,128,256),cv::Scalar(0,256,256),cv::Scalar(256,0,256),
		cv::Scalar(256,256,128),cv::Scalar(128,0,256),cv::Scalar(128,128,128),cv::Scalar(128,0,128),cv::Scalar(128,256,0),cv::Scalar(256,256,256),cv::Scalar(256,128,0),
		cv::Scalar(0,128,128),cv::Scalar(0,0,256),cv::Scalar(256,128,128),cv::Scalar(0,256,128),cv::Scalar(128,128,0),cv::Scalar(256,128,256),cv::Scalar(0,128,256),
		cv::Scalar(128,256,256),cv::Scalar(256,0,0),cv::Scalar(256,256,0),cv::Scalar(0,128,0),cv::Scalar(0,256,0),cv::Scalar(256,0,128),cv::Scalar(128,256,128) };

	// pose object
	pose_points pose;

public:

	Estimator();

	int init(int& outCameraWidth, int& outCameraHeight, int detectRatio, int camId, float fovZoom, bool draw);

	void close();

	void detect(TransformData& outFaces, ExpressionData* outExpression);

	cv::Mat GetLineFace(dlib::full_object_detection landmarks);

	void getRawImageBytes(unsigned char* data, int width, int height);

};


