/*------------------------------------------------------------------------------------------*\
   This file contains material supporting chapter 7 of the cookbook:  
   Computer Vision Programming using the OpenCV Library 
   Second Edition 
   by Robert Laganiere, Packt Publishing, 2013.

   This program is free software; permission is hereby granted to use, copy, modify, 
   and distribute this source code, or portions thereof, for any purpose, without fee, 
   subject to the restriction that the copyright notice may not be removed 
   or altered from any source or altered source distribution. 
   The software is released on an as-is basis and without any warranties of any kind. 
   In particular, the software is not guaranteed to be fault-tolerant or free from failure. 
   The author disclaims all warranties with regard to this software, any use, 
   and any consequent failure, is purely the responsibility of the user.
 
   Copyright (C) 2013 Robert Laganiere, www.laganiere.name
\*------------------------------------------------------------------------------------------*/

#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include "harrisDetector.h"

int main()
{
	// Harris:

	// Read input image
	cv::Mat image= cv::imread("church01.jpg",0);
	if (!image.data)
		return 0; 

    // Display the image
	cv::namedWindow("Original Image");
	cv::imshow("Original Image",image);

	// Detect Harris Corners
	cv::Mat cornerStrength;
	cv::cornerHarris(image,cornerStrength,
		             3,     // neighborhood size
					 3,     // aperture size
					 0.01); // Harris parameter

   // threshold the corner strengths
	cv::Mat harrisCorners;
	double threshold= 0.0001; 
	cv::threshold(cornerStrength,harrisCorners,
                 threshold,255,cv::THRESH_BINARY_INV);

    // Display the corners
	cv::namedWindow("Harris Corner Map");
	cv::imshow("Harris Corner Map",harrisCorners);

	// Create Harris detector instance
	HarrisDetector harris;
    // Compute Harris values
	harris.detect(image);
    // Detect Harris corners
	std::vector<cv::Point> pts;
	harris.getCorners(pts,0.01);
	// Draw Harris corners
	harris.drawOnImage(image,pts);

    // Display the corners
	cv::namedWindow("Harris Corners");
	cv::imshow("Harris Corners",image);

	// GFTT:

	// Read input image
	image= cv::imread("church01.jpg",0);

	// Compute good features to track
	std::vector<cv::Point2f> corners;
	cv::goodFeaturesToTrack(image,corners,
		500,	// maximum number of corners to be returned
		0.01,	// quality level
		10);	// minimum allowed distance between points
	  
	// for all corners
	std::vector<cv::Point2f>::const_iterator it= corners.begin();
	while (it!=corners.end()) {

		// draw a circle at each corner location
		cv::circle(image,*it,3,cv::Scalar(255,255,255),2);
		++it;
	}

    // Display the corners
	cv::namedWindow("Good Features to Track");
	cv::imshow("Good Features to Track",image);

	// Read input image
	image= cv::imread("church01.jpg",0);

	// vector of keypoints
	std::vector<cv::KeyPoint> keypoints;
	// Construction of the Good Feature to Track detector 
	cv::GoodFeaturesToTrackDetector gftt(
		500,	// maximum number of corners to be returned
		0.01,	// quality level
		10);	// minimum allowed distance between points
	// point detection using FeatureDetector method
	gftt.detect(image,keypoints);
	
	cv::drawKeypoints(image,		// original image
		keypoints,					// vector of keypoints
		image,						// the resulting image
		cv::Scalar(255,255,255),	// color of the points
		cv::DrawMatchesFlags::DRAW_OVER_OUTIMG); //drawing flag

    // Display the corners
	cv::namedWindow("Good Features to Track Detector");
	cv::imshow("Good Features to Track Detector",image);

	// FAST feature:

	// Read input image
	image= cv::imread("church01.jpg",0);

	keypoints.clear();
	cv::FastFeatureDetector fast(40);
	fast.detect(image,keypoints);
	
	cv::drawKeypoints(image,keypoints,image,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
	std::cout << "Number of keypoints (FAST): " << keypoints.size() << std::endl; 

    // Display the corners
	cv::namedWindow("FAST Features");
	cv::imshow("FAST Features",image);

	// FAST feature without non-max suppression
	// Read input image
	image= cv::imread("church01.jpg",0);

	keypoints.clear();
	cv::FAST(image,     // input image 
		     keypoints, // output vector of keypoints
			 40,        // threshold
			 false);    // non-max suppression (or not)
	
	cv::drawKeypoints(image,keypoints,image,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

    // Display the corners
	cv::namedWindow("FAST Features (all)");
	cv::imshow("FAST Features (all)",image);

	// Dynamically adapted FAST feature
	// Read input image
	image= cv::imread("church01.jpg",0);

	keypoints.clear();
	cv::DynamicAdaptedFeatureDetector fastD(
		new cv::FastAdjuster(40), // the feature detector
		150,   // min number of features
		200,   // max number of features
		50);   // max number of iterations
//	or
//	cv::DynamicAdaptedFeatureDetector fastD(cv::FastAdjuster::create("FAST"),150,200,50);

	fastD.detect(image,keypoints); // detect points
	std::cout << "Number of keypoints (should be between 150 and 200): " << keypoints.size() << std::endl; 
	
	cv::drawKeypoints(image,keypoints,image,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

    // Display the corners
	cv::namedWindow("FAST Features [150,200]");
	cv::imshow("FAST Features [150,200]",image);

	// Grid adapted FAST feature
	// Read input image
	image= cv::imread("church01.jpg",0);

	keypoints.clear();
	cv::GridAdaptedFeatureDetector fastG(
		new cv::FastFeatureDetector(10), // the feature detector
		1200,   // max total number of keypoints
		5,     // number of rows in grid
		2);    // number of cols in grid

	fastG.detect(image,keypoints);
	std::cout << "Number of keypoints in grid (should be around 1200): " << keypoints.size() << std::endl; 
	
	cv::drawKeypoints(image,keypoints,image,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);

    // Display the corners
	cv::namedWindow("FAST Features (5x2 grid)");
	cv::imshow("FAST Features (5x2 grid)",image);

	// Pyramid adapted FAST feature
	// Read input image
	image= cv::imread("church01.jpg",CV_LOAD_IMAGE_GRAYSCALE);

	keypoints.clear();
	cv::PyramidAdaptedFeatureDetector fastP(
		new cv::FastFeatureDetector(80), // the feature detector
		3);    // number of levels

	fastP.detect(image,keypoints);
	std::cout << "keypoint: " << keypoints[1].class_id << " , " << keypoints[1].octave << " , " << keypoints[1].size << std::endl; 
	std::cout << "keypoint: " << keypoints[111].class_id << " , " << keypoints[111].octave << " , " << keypoints[111].size << std::endl; 
	std::cout << "keypoint: " << keypoints[keypoints.size()-2].class_id << " , " << keypoints[keypoints.size()-2].octave << " , " << keypoints[keypoints.size()-2].size << std::endl; 
	
	cv::drawKeypoints(image,keypoints,image,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Display the corners
	cv::namedWindow("FAST Features (3 levels)");
	cv::imshow("FAST Features (3 levels)",image);

	// SURF:

	// Read input image
	image= cv::imread("church01.jpg",CV_LOAD_IMAGE_GRAYSCALE);

	keypoints.clear();
	// Construct the SURF feature detector object
    //	cv::SURF surf(3000.0);
	// Detect the SURF features
    //	surf(image,cv::Mat(),keypoints);

	// Construct the SURF feature detector object
	cv::Ptr<cv::FeatureDetector> detector = new cv::SURF(3000.);
	// Detect the SURF features
	detector->detect(image,keypoints);
	
	cv::Mat featureImage;
	cv::drawKeypoints(image,keypoints,featureImage,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Display the corners
	cv::namedWindow("SURF Features");
	cv::imshow("SURF Features",featureImage);

	std::cout << "Number of SURF keypoints: " << keypoints.size() << std::endl; 

	// Read a second input image
	image= cv::imread("church03.jpg",CV_LOAD_IMAGE_GRAYSCALE);

	// Detect the SURF features
	detector->detect(image,keypoints);
	
	cv::drawKeypoints(image,keypoints,featureImage,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Display the corners
	cv::namedWindow("SURF Features at another scale");
	cv::imshow("SURF Features at another scale",featureImage);

	// SIFT:

	// Read input image
	image= cv::imread("church01.jpg",CV_LOAD_IMAGE_GRAYSCALE);

	keypoints.clear();

	// Construct the SIFT feature detector object
	detector = new cv::SIFT();
	// Detect the SIFT features
	detector->detect(image,keypoints);
	
	cv::drawKeypoints(image,keypoints,featureImage,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Display the corners
	cv::namedWindow("SIFT Features");
	cv::imshow("SIFT Features",featureImage);

	std::cout << "Number of SIFT keypoints: " << keypoints.size() << std::endl; 

	// BRISK:

	// Read input image
	image= cv::imread("church01.jpg",CV_LOAD_IMAGE_GRAYSCALE);

	keypoints.clear();

	// Construct the BRISK feature detector object
	detector = new cv::BRISK();
	// Detect the BRISK features
	detector->detect(image,keypoints);
	
	cv::drawKeypoints(image,keypoints,featureImage,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Display the corners
	cv::namedWindow("BRISK Features");
	cv::imshow("BRISK Features",featureImage);

	// Construct another BRISK feature detector object
	detector = new cv::BRISK(
		20,  // threshold for FAST points to be accepted
		5);  // number of octaves

	// Detect the BRISK features
	detector->detect(image,keypoints);
	
	cv::drawKeypoints(image,keypoints,featureImage,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Display the corners
	cv::namedWindow("BRISK Features (2)");
	cv::imshow("BRISK Features (2)",featureImage);

	std::cout << "Number of BRISK keypoints: " << keypoints.size() << std::endl; 

	// ORB:

	// Read input image
	image= cv::imread("church01.jpg",CV_LOAD_IMAGE_GRAYSCALE);

	keypoints.clear();

	// Construct the ORB feature detector object
	detector = new cv::ORB(200, // total number of keypoints
		                   1.2, // scale factor between layers
						   8);  // number of layers in pyramid
	// Detect the ORB features
	detector->detect(image,keypoints);
	
	cv::drawKeypoints(image,keypoints,featureImage,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Display the corners
	cv::namedWindow("ORB Features");
	cv::imshow("ORB Features",featureImage);

	std::cout << "Number of ORB keypoints: " << keypoints.size() << std::endl; 

	/*
	cv::Rect neighbors(10,10,3,3);
	cv::Mat cornerA(image,neighbors);
	neighbors.x=20;
	neighbors.y=20;
	cv::Mat cornerB(image,neighbors);
	cv::Mat result;
	result=0;
	cv::matchTemplate(cornerA,cornerB,result,CV_TM_SQDIFF);
	std::cout << cornerA << std::endl;
	std::cout << cornerB << std::endl;
	std::cout << result << std::endl;


	
	// image matching

	// 1. Read input images
	cv::Mat image1= cv::imread("church01.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat image2= cv::imread("church02.jpg",CV_LOAD_IMAGE_GRAYSCALE);

	// 2. Define keypoints vector
	std::vector<cv::KeyPoint> keypoints1;
	std::vector<cv::KeyPoint> keypoints2;

	// 3. Define feature detector
	cv::FastFeatureDetector fastDet(80);

	// 4. Keypoint detection
	fastDet.detect(image1,keypoints1);
	fastDet.detect(image1,keypoints2);

	// 5. Define a neighborhood
	cv::Rect neighbors(0,0,3,3); // 3x3
	cv::Mat patch1;
	cv::Mat patch2;

	// 5. Forall keypoints in first image
	//    find best match in second image
	cv::Mat result;
	int best;

	//for all keypoints in image 1
	std::vector<cv::KeyPoint>::iterator it1= keypoints1.begin();
	while (it1!= keypoints1.end()) {
	
		// define image patch
		neighbors.x= it1->pt.x;
		neighbors.x= it1->pt.x;
		patch1= image1(neighbors);

		// reset best correlation value;
		best= 1000000;

		//for all keypoints in image 2
		std::vector<cv::KeyPoint>::iterator it2= keypoints2.begin();
		while (it2!= keypoints2.end()) {

			// define image patch
			neighbors.x= it2->pt.x;
			neighbors.x= it2->pt.x;
			patch2= image2(neighbors);
			cv::matchTemplate(patch1,patch2,result,CV_TM_SQDIFF);

			// check if it is a best match
			if (result.at<uchar>(0,0)<best) {

			}

			it2++;
		}
			
		it1++;
	}
	*/
	/*
	// Read input image
	image= cv::imread("church03.jpg",0);

	keypoints.clear();
	// Construct the SURF feature detector object
	cv::SurfFeatureDetector surf(2500);
	// Detect the SURF features
	surf.detect(image,keypoints);
	
	cv::Mat featureImage;
	cv::drawKeypoints(image,keypoints,featureImage,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Display the corners
	cv::namedWindow("SURF Features");
	cv::imshow("SURF Features",featureImage);

	// Read input image
	image= cv::imread("church01.jpg",0);

	keypoints.clear();
	// Construct the SURF feature detector object
	cv::SiftFeatureDetector sift(
		0.03,  // feature threshold
		10.);  // threshold to reduce
	           // sensitivity to lines

	// Detect the SURF features
	sift.detect(image,keypoints);
	
	cv::drawKeypoints(image,keypoints,featureImage,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Display the corners
	cv::namedWindow("SIFT Features");
	cv::imshow("SIFT Features",featureImage);

	// Read input image
	image= cv::imread("church01.jpg",0);

	keypoints.clear();

	cv::MserFeatureDetector mser;
	mser.detect(image,keypoints);
	
	// Draw the keypoints with scale and orientation information
	cv::drawKeypoints(image,		// original image
		keypoints,					// vector of keypoints
		featureImage,				// the resulting image
		cv::Scalar(255,255,255),	// color of the points
		cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS); //drawing flag

    // Display the corners
	cv::namedWindow("MSER Features");
	cv::imshow("MSER Features",featureImage);
	*/
	cv::waitKey();
	return 0;
}