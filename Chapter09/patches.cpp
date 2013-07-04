/*------------------------------------------------------------------------------------------*\
   This file contains material supporting chapter 9 of the cookbook:  
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

int main()
{
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
	fastDet.detect(image2,keypoints2);

	// 5. Define a square neighborhood
	cv::Rect neighbors(0,0,11,11); // 11x11
	cv::Mat patch1;
	cv::Mat patch2;

	// 6. Forall keypoints in first image
	//    find best match in second image
	cv::Mat result;
	std::vector<cv::DMatch> matches;

	//for all keypoints in image 1
	for (int i=0; i<keypoints1.size(); i++) {
	
		// define image patch
		neighbors.x= keypoints1[i].pt.x-5;
		neighbors.y= keypoints1[i].pt.y-5;

		// if neighborhood of points outside image, then continue with next point
		if (neighbors.x<0 || neighbors.y<0 || 
			neighbors.x+11 >= image1.cols || neighbors.y+11 >= image1.rows)
			continue;

		//patch in image 1
		patch1= image1(neighbors);

		// reset best correlation value;
		cv::DMatch bestMatch;

		//for all keypoints in image 2
	    for (int j=0; j<keypoints2.size(); j++) {

			// define image patch
			neighbors.x= keypoints2[j].pt.x-5;
			neighbors.y= keypoints2[j].pt.y-5;

			// if neighborhood of points outside image, then continue with next point
			if (neighbors.x<0 || neighbors.y<0 || 
				neighbors.x+11 >= image2.cols || neighbors.y+11 >= image2.rows)
				continue;

			// patch in image 2
			patch2= image2(neighbors);

			// match the two patches
			cv::matchTemplate(patch1,patch2,result,CV_TM_SQDIFF_NORMED);

			// check if it is a best match
			if (result.at<float>(0,0) < bestMatch.distance) {

				bestMatch.distance= result.at<float>(0,0);
				bestMatch.queryIdx= i;
				bestMatch.trainIdx= j;
			}
		}

		// add the best match
		matches.push_back(bestMatch);
	}

	// extract the 25 best matches
	std::nth_element(matches.begin(),matches.begin()+25,matches.end());
	matches.erase(matches.begin()+25,matches.end());

	// Draw the matching results
	cv::Mat matchImage;
	cv::drawMatches(image1,keypoints1, // first image
                   image2,keypoints2, // second image
                   matches,     // vector of matches
                   matchImage,  // produced image
	               cv::Scalar(255,255,255),  // line color
				   cv::Scalar(255,255,255)); // point color

    // Display the image of matches
	cv::namedWindow("Matches");
	cv::imshow("Matches",matchImage);

	// Match template

	// define a template
	cv::Mat target(image1,cv::Rect(80,105,30,30));
    // Display the template
	cv::namedWindow("Template");
	cv::imshow("Template",target);

	// define search region
	cv::Mat roi(image2, 
		// here top half of the image
		cv::Rect(0,0,image2.cols,image2.rows/2)); 
			
	// perform template matching
	cv::matchTemplate(
		roi,    // search region
		target, // template
		result, // result
		CV_TM_SQDIFF); // similarity measure

	// find most similar location
	double minVal, maxVal;
	cv::Point minPt, maxPt;
	cv::minMaxLoc(result, &minVal, &maxVal, &minPt, &maxPt);

	// draw rectangle at most similar location
	// at minPt in this case
	cv::rectangle(roi, cv::Rect(minPt.x, minPt.y, target.cols , target.rows), 255);
	
    // Display the template
	cv::namedWindow("Best");
	cv::imshow("Best",image2);

	cv::waitKey();
	return 0;
}