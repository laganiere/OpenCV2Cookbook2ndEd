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

#include "histogram.h"


// Create an image representing a histogram
cv::Mat Histogram1D::getImageOfHistogram(const cv::Mat &hist, int zoom) {

	// Get min and max bin values
	double maxVal = 0;
	double minVal = 0;
	cv::minMaxLoc(hist, &minVal, &maxVal, 0, 0);

	// get histogram size
	int histSize = hist.rows;

	// Square image on which to display histogram
	cv::Mat histImg(histSize*zoom, histSize*zoom, CV_8U, cv::Scalar(255));

	// set highest point at 90% of nbins (i.e. image height)
	int hpt = static_cast<int>(0.9*histSize);

	// Draw vertical line for each bin 
	for (int h = 0; h < histSize; h++) {

		float binVal = hist.at<float>(h);
		if (binVal>0) {
			int intensity = static_cast<int>(binVal*hpt / maxVal);
			cv::line(histImg, cv::Point(h*zoom, histSize*zoom),
				cv::Point(h*zoom, (histSize - intensity)*zoom), cv::Scalar(0), zoom);
		}
	}

	return histImg;
}
