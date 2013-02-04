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

template <typename T, int N>
class IntegralImage {

	  cv::Mat integralImage;

  public:

	  IntegralImage(cv::Mat image) {

		cv::integral(image,integralImage,cv::DataType<T>::type);
	  }

	  cv::Vec<T,N> operator()(int xo, int yo, int width, int height) {

		  return (integralImage.at<cv::Vec<T,N>>(yo+height,xo+width)
			      -integralImage.at<cv::Vec<T,N>>(yo+height,xo)
			      -integralImage.at<cv::Vec<T,N>>(yo,xo+width)
			      +integralImage.at<cv::Vec<T,N>>(yo,xo));
	  }

	  cv::Vec<T,N> operator()(int x, int y, int radius) {

		  return (integralImage.at<cv::Vec<T,N>>(y+radius+1,x+radius+1)
			      -integralImage.at<cv::Vec<T,N>>(y+radius+1,x-radius)
			      -integralImage.at<cv::Vec<T,N>>(y-radius,x+radius+1)
			      +integralImage.at<cv::Vec<T,N>>(y-radius,x-radius));
	  }
};

int main()
{
	cv::Mat image= cv::imread("bike55.bmp",0);
	if (!image.data)
		return 0; 

	cv::Mat roi(image,cv::Rect(97,112,25,30));

	// compute histogram of 16 bins
	Histogram1D h;
	h.setNBins(16);
	cv::Mat histo= h.getHistogram(roi);

	cv::namedWindow("Histo");
	cv::imshow("Histo",h.getHistogramImage(roi,16));

	// Loop over each bin
	for (int i=0; i<16; i++) 
		std::cout << "Value " << i << " = " << histo.at<float>(i) << std::endl;  

	// create vector of 16 binay images
	std::vector<cv::Mat> planes;
	image= image&0xF0;

	for (int i=0; i<16; i++) {

		// 1 for each pixel equals to i<<4
		planes.push_back((image==(i<<4))&0x1);
	}

	// create 16-channel image
	cv::Mat dst;
	cv::merge(planes,dst);

	// compute integral image
	cv::Mat sum;
	cv::integral(dst,sum);

	IntegralImage<int,16> integral(dst);
	cv::Vec<int,16> hh= integral(97,112,25,30);
	std::cout<<"valeur="<<hh<<std::endl;

	cv::Mat hi(hh),ho;
	hi.convertTo(ho,CV_32F);
	std::cout<<"valeur="<<ho;
	h.setNBins(16);

	cv::namedWindow("Histo2");
	cv::Mat im= h.getImageOfHistogram(ho,16);
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