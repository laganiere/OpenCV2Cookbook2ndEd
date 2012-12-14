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

#include "colordetector.h"

int main()
{
    // 1. Create image processor object
	ColorDetector cdetect;

    // 2. Read input image
	cv::Mat image= cv::imread("boldt.jpg");
	if (!image.data)
		return 0; 

    // 3. Set input parameters
	cdetect.setTargetColor(230,190,130); // here blue sky

   // 4. Process the image and display the result
	cv::namedWindow("result");
	cv::imshow("result",cdetect.process(image));

	// or using functor
	ColorDetector colordetector(230,190,130,  // color
		                                100); // threshold
	cv::namedWindow("result (functor)");
	cv::imshow("result (functor)",colordetector(image));

	cv::waitKey();

	return 0;
}

