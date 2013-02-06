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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

#include "histogram.h"
#include "integral.h"

int main()
{
	// Open image
	cv::Mat image= cv::imread("bike55.bmp",0);
	// define image roi
	int xo=97, yo=112;
	int width=25, height=30;
	cv::Mat roi(image,cv::Rect(xo,yo,width,height));
	// compute sum
	// returns a Scalar to work with multi-channel images
	cv::Scalar sum= cv::sum(roi);

	std::cout << sum[0] << std::endl;

	// compute integral image
	cv::Mat integralImage;
	cv::integral(image,integralImage,CV_32S);
	// get sum over an area using three additions/subtractions
    int sumInt= integralImage.at<int>(yo+height,xo+width)
			      -integralImage.at<int>(yo+height,xo)
			      -integralImage.at<int>(yo,xo+width)
			      +integralImage.at<int>(yo,xo);

	std::cout << sumInt << std::endl;

	// compute histogram of 16 bins
	Histogram1D h;
	h.setNBins(16);
	cv::Mat histo= h.getHistogram(roi);

	cv::namedWindow("Histo");
	cv::imshow("Histo",h.getHistogramImage(roi,16));
	std::cout << histo << std::endl;

	// compute histogram of 16 bins with integral image
	cv::Mat planes;
	convertToBinaryPlanes(image,planes,16);
	IntegralImage<float,16> intHisto(planes);
	cv::Vec<float,16> histogram= intHisto(97,112,25,30);
	std::cout<< histogram << std::endl;

	cv::namedWindow("Histo2");
	cv::Mat im= h.getImageOfHistogram(cv::Mat(histogram),16);
	cv::imshow("Histo2",im);	
/*
	cv::Mat image2= cv::imread("bike65.bmp",0);
	if (!image2.data)
		return 0; 

	// reduce to 16 gray shades
	image2= image2&0xF0;

	cv::Mat roi2(image2,cv::Rect(135,114,25,30));

	Histogram1D h2;
	cv::namedWindow("Histo 2");
	cv::imshow("Histo 2",h2.getHistogramImage(roi2));

	cv::rectangle(image2,cv::Rect(135,114,25,30),cv::Scalar(0,0,255));
	cv::namedWindow("image 2");
	cv::imshow("image 2",image2);
	*/
	cv::waitKey();
	
}