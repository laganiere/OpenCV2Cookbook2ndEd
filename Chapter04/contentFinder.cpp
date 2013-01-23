/*------------------------------------------------------------------------------------------*\
   This file contains material supporting chapter 4 of the cookbook:  
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
using namespace std;

#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>

#include "histogram.h"
#include "contentFinder.h"
#include "colorhistogram.h"

int main()
{
	// Read input image
	cv::Mat image= cv::imread("waves.jpg",0);
	if (!image.data)
		return 0; 

	// define image ROI
	cv::Mat imageROI;
	imageROI= image(cv::Rect(360,55,40,50)); // Cloud region

	// Display reference patch
	cv::namedWindow("Reference");
	cv::imshow("Reference",imageROI);

	// Find histogram of reference
	Histogram1D h;
	cv::Mat hist= h.getHistogram(imageROI);
	cv::namedWindow("Reference Hist");
	cv::imshow("Reference Hist",h.getHistogramImage(imageROI));

	// Create the content finder
	ContentFinder finder;
	finder.setHistogram(hist);

	finder.setThreshold(-1.0f);

	// Get back-projection
	cv::Mat result1;
	result1= finder.find(image);

	// Create negative image and display result
	cv::Mat tmp;
	result1.convertTo(tmp,CV_8U,-1.0,255.0);
	cv::namedWindow("Backprojection result");
	cv::imshow("Backprojection result",tmp);

	// Get binary back-projection
	finder.setThreshold(0.12f);
	result1= finder.find(image);

	// Draw a rectangle around the reference area
	cv::rectangle(image,cv::Rect(360,55,40,50),cv::Scalar(0,0,0));

	// Display image
	cv::namedWindow("Image");
	cv::imshow("Image",image);

	// Display result
	cv::namedWindow("Detection Result");
	cv::imshow("Detection Result",result1);

	// Load color image
	ColorHistogram hc;
	cv::Mat color= cv::imread("waves.jpg");
	color= hc.colorReduce(color,32);

	// Draw a rectangle around the reference area
	// cv::rectangle(color,cv::Rect(0,0,165,75),cv::Scalar(0,0,0));

	cv::namedWindow("Color Image");
	cv::imshow("Color Image",color);

	imageROI= color(cv::Rect(0,0,165,75)); // blue sky area

	// Get 3D colour histogram
	cv::Mat shist= hc.getHistogram(imageROI);

	finder.setHistogram(shist);
	finder.setThreshold(0.05f);

	// Get back-projection of colour histogram
	result1= finder.find(color);

	cv::namedWindow("Color Detection Result");
	cv::imshow("Color Detection Result",result1);
/*
	// Second colour image
	cv::Mat colour2= cv::imread("../dog.jpg");
	colour2= hc.colorReduce(colour2,32);

	// Get back-projection of colour histogram
	result2= finder.find(colour2);

	cv::namedWindow("Result colour (2)");
	cv::imshow("Result colour (2)",result2);

	// Get ab colour histogram
	colour= cv::imread("../waves.jpg");
	imageROI= colour(cv::Rect(0,0,165,75)); // blue sky area
	cv::Mat colourhist= hc.getabHistogram(imageROI);

	finder.setHistogram(colourhist);
	finder.setThreshold(0.05f);

	// Convert to Lab space
	cv::Mat lab;
	cv::cvtColor(colour, lab, CV_BGR2Lab);

	// Get back-projection of ab histogram
	int ch[2]={1,2};
	result1= finder.find(lab,-128.0f,127.0f,ch,2);

	cv::namedWindow("Result ab (1)");
	cv::imshow("Result ab (1)",result1);

	// Second colour image
	colour2= cv::imread("../dog.jpg");

	cv::namedWindow("Colour Image (2)");
	cv::imshow("Colour Image (2)",colour2);

	cv::cvtColor(colour2, lab, CV_BGR2Lab);

	result2= finder.find(lab,-128.0f,127.0f,ch,2);

	cv::namedWindow("Result ab (2)");
	cv::imshow("Result ab (2)",result2);

	// Get Hue colour histogram
	colour= cv::imread("waves.jpg");
	imageROI= colour(cv::Rect(0,0,165,75)); // blue sky area
	colourhist= hc.getHueHistogram(imageROI);

	finder.setHistogram(colourhist);
	finder.setThreshold(0.3f);

	// Convert to HSV space
	cv::Mat hsv;
	cv::cvtColor(colour, hsv, CV_BGR2HSV);

	// Get back-projection of hue histogram
	ch[0]=0;
	result1= finder.find(hsv,0.0f,180.0f,ch,1);

	cv::namedWindow("Result Hue (1)");
	cv::imshow("Result Hue (1)",result1);

	// Second colour image
	colour2= cv::imread("../dog.jpg");

	cv::cvtColor(colour2, hsv, CV_BGR2HSV);

	result2= finder.find(hsv,0.0f,180.0f,ch,1);

	cv::namedWindow("Result Hue (2)");
	cv::imshow("Result Hue (2)",result2);
*/
	cv::waitKey();
	return 0;
}

