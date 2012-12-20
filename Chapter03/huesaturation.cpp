/*------------------------------------------------------------------------------------------*\
   This file contains material supporting chapter 3 of the cookbook:  
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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <vector>

int main()
{

	cv::Mat image= cv::imread("boldt.jpg");
	if (!image.data)
		return 0; 

	cv::namedWindow("Original image");
	cv::imshow("Original image",image);

	cv::Mat hsv;
	cv::cvtColor(image, hsv, CV_BGR2HSV);

	std::vector<cv::Mat> channels;
	cv::split(hsv,channels);

	cv::namedWindow("Value");
	cv::imshow("Value",channels[2]);

	cv::namedWindow("Saturation");
	cv::imshow("Saturation",channels[1]);

	cv::namedWindow("Hue");
	cv::imshow("Hue",channels[0]);

	cv::Mat satMask;
	cv::threshold(channels[1], satMask, 40, 255, cv::THRESH_BINARY_INV);

	cv::add(channels[0], satMask, channels[0]);
	cv::namedWindow("Hue with mask");
	cv::imshow("Hue with mask",channels[0]);

	cv::Mat valMask;
	cv::threshold(channels[2], valMask, 20, 255, cv::THRESH_BINARY_INV);

	cv::add(channels[0], valMask, channels[0]);
	cv::namedWindow("Hue with 2mask");
	cv::imshow("Hue with 2mask",channels[0]);

	cv::waitKey();
}
