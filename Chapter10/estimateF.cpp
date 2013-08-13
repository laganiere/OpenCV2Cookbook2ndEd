/*------------------------------------------------------------------------------------------*\
   This file contains material supporting chapter 10 of the cookbook:  
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
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

int main()
{
	// Read input images
	cv::Mat image1= cv::imread("church01.jpg",0);
	cv::Mat image2= cv::imread("church03.jpg",0);
	if (!image1.data || !image2.data)
		return 0; 

    // Display the images
	cv::namedWindow("Right Image");
	cv::imshow("Right Image",image1);
	cv::namedWindow("Left Image");
	cv::imshow("Left Image",image2);

	// vector of keypoints
	std::vector<cv::KeyPoint> keypoints1;
	std::vector<cv::KeyPoint> keypoints2;

	// Construction of the SURF feature detector 
	cv::SURF surf(1000);

	// Detection of the SURF features
	surf.detect(image1,keypoints1);
	surf.detect(image2,keypoints2);

	std::cout << "Number of SURF points (1): " << keypoints1.size() << std::endl;
	std::cout << "Number of SURF points (2): " << keypoints2.size() << std::endl;
	
	// Draw the kepoints
	cv::Mat imageKP;
	cv::drawKeypoints(image1,keypoints1,imageKP,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::namedWindow("Right SURF Features");
	cv::imshow("Right SURF Features",imageKP);
	cv::drawKeypoints(image2,keypoints2,imageKP,cv::Scalar(255,255,255),cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::namedWindow("Left SURF Features");
	cv::imshow("Left SURF Features",imageKP);

	// Extraction of the SURF descriptors
	cv::Mat descriptors1, descriptors2;
	surf.compute(image1,keypoints1,descriptors1);
	surf.compute(image2,keypoints2,descriptors2);

	// Construction of the matcher 
	cv::BFMatcher matcher(cv::NORM_L2,true);

	// Match the two image descriptors
	std::vector<cv::DMatch> matches;
	matcher.match(descriptors1,descriptors2, matches);

	std::cout << "Number of matched points: " << matches.size() << std::endl;

	// Select few Matches  
	keypoints1.push_back(cv::KeyPoint(15,189,1));
	keypoints1.push_back(cv::KeyPoint(140,242,1));
	keypoints2.push_back(cv::KeyPoint(48,182,1));
	keypoints2.push_back(cv::KeyPoint(118,222,1));
	std::vector<cv::DMatch> selMatches;

	/* between church01 and church03 */
	selMatches.push_back(matches[1]);  
	selMatches.push_back(matches[8]);  
	selMatches.push_back(matches[0]);  
	selMatches.push_back(matches[16]);  
	selMatches.push_back(matches[20]);  
	selMatches.push_back(cv::DMatch(keypoints1.size()-2,keypoints2.size()-2,1));
	selMatches.push_back(cv::DMatch(keypoints1.size()-1,keypoints2.size()-1,1));

	// Draw the selected matches
	cv::Mat imageMatches;
	cv::drawMatches(image1,keypoints1,  // 1st image and its keypoints
		            image2,keypoints2,  // 2nd image and its keypoints
					selMatches,			// the matches
//					matches,			// the matches
					imageMatches,		// the image produced
					cv::Scalar(255,255,255),
					cv::Scalar(255,255,255),
					std::vector<char>(),
					2
					); // color of the lines
	cv::namedWindow("Matches");
	cv::imshow("Matches",imageMatches);

	// Convert 1 vector of keypoints into
	// 2 vectors of Point2f
	std::vector<int> pointIndexes1;
	std::vector<int> pointIndexes2;
	for (std::vector<cv::DMatch>::const_iterator it= selMatches.begin();
		 it!= selMatches.end(); ++it) {

			 // Get the indexes of the selected matched keypoints
			 pointIndexes1.push_back(it->queryIdx);
			 pointIndexes2.push_back(it->trainIdx);
	}
		 
	// Convert keypoints into Point2f
	std::vector<cv::Point2f> selPoints1, selPoints2;
	cv::KeyPoint::convert(keypoints1,selPoints1,pointIndexes1);
	cv::KeyPoint::convert(keypoints2,selPoints2,pointIndexes2);

	// check by drawing the points 
	std::vector<cv::Point2f>::const_iterator it= selPoints1.begin();
	while (it!=selPoints1.end()) {

		// draw a circle at each corner location
		cv::circle(image1,*it,3,cv::Scalar(255,255,255),2);
		++it;
	}

	it= selPoints2.begin();
	while (it!=selPoints2.end()) {

		// draw a circle at each corner location
		cv::circle(image2,*it,3,cv::Scalar(255,255,255),2);
		++it;
	}

	// Compute F matrix from 7 matches
	cv::Mat fundemental= cv::findFundamentalMat(
		selPoints1, // points in first image
		selPoints2, // points in second image
		CV_FM_7POINT);       // 7-point method

	std::cout << "F-Matrix size= " << fundemental.rows << "," << fundemental.cols << std::endl;  
	cv::Mat fund(fundemental,cv::Rect(0,0,3,3));
	// draw the left points corresponding epipolar lines in right image 
	std::vector<cv::Vec3f> lines1; 
	cv::computeCorrespondEpilines(
		selPoints1, // image points 
		1,                   // in image 1 (can also be 2)
		fund, // F matrix
		lines1);     // vector of epipolar lines

	// for all epipolar lines
	for (std::vector<cv::Vec3f>::const_iterator it= lines1.begin();
		 it!=lines1.end(); ++it) {

			 // draw the epipolar line between first and last column
			 cv::line(image2,cv::Point(0,-(*it)[2]/(*it)[1]),
				             cv::Point(image2.cols,-((*it)[2]+(*it)[0]*image2.cols)/(*it)[1]),
							 cv::Scalar(255,255,255));
	}
		
	// draw the left points corresponding epipolar lines in left image 
	std::vector<cv::Vec3f> lines2;
	cv::computeCorrespondEpilines(cv::Mat(selPoints2),2,fund,lines2);
	for (std::vector<cv::Vec3f>::const_iterator it= lines2.begin();
		 it!=lines2.end(); ++it) {

			 // draw the epipolar line between first and last column
			 cv::line(image1,cv::Point(0,-(*it)[2]/(*it)[1]),
				             cv::Point(image1.cols,-((*it)[2]+(*it)[0]*image1.cols)/(*it)[1]),
							 cv::Scalar(255,255,255));
	}
		
    // Display the images with points and epipolar lines
	cv::namedWindow("Epilines (1)");
	cv::imshow("Epilines (1)",image1);
	cv::namedWindow("Epilines (2)");
	cv::imshow("Epilines (2)",image2);

	cv::waitKey();
	return 0;
}