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

#if !defined COLHISTOGRAM
#define COLHISTOGRAM

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>

class ColorHistogram {

  private:

    int histSize[3];        // size of each dimension
	float hranges[2];       // range of values
    const float* ranges[3]; // array of ranges for each dimension
    int channels[3];        // channel to be considered

  public:

	ColorHistogram() {

		// Prepare arguments for a color histogram
		// each dimension having size of 256
		histSize[0]= histSize[1]= histSize[2]= 256;
		hranges[0]= 0.0;    // BRG range from 0 to 255
		hranges[1]= 255.0;
		ranges[0]= hranges; // all channels have the same range 
		ranges[1]= hranges; 
		ranges[2]= hranges; 
		channels[0]= 0;		// the three channels 
		channels[1]= 1; 
		channels[2]= 2; 
	}

	// Computes the histogram.
	cv::Mat getHistogram(const cv::Mat &image) {

		cv::Mat hist;

		// BGR color histogram
		hranges[0]= 0.0;    // BRG range
		hranges[1]= 255.0;
		channels[0]= 0;		// the three channels 
		channels[1]= 1; 
		channels[2]= 2; 

		// Compute histogram
		cv::calcHist(&image, 
			1,			// histogram of 1 image only
			channels,	// the channel used
			cv::Mat(),	// no mask is used
			hist,		// the resulting histogram
			3,			// it is a 3D histogram
			histSize,	// number of bins
			ranges		// pixel value range
		);

		return hist;
	}

	// Computes the histogram.
	cv::SparseMat getSparseHistogram(const cv::Mat &image) {

		cv::SparseMat hist(3,        // number of dimensions
			               histSize, // size of each dimension
						   CV_32F);


		// BGR color histogram
		hranges[0]= 0.0;    // BRG range
		hranges[1]= 255.0;
		channels[0]= 0;		// the three channels 
		channels[1]= 1; 
		channels[2]= 2; 

		// Compute histogram
		cv::calcHist(&image, 
			1,			// histogram of 1 image only
			channels,	// the channel used
			cv::Mat(),	// no mask is used
			hist,		// the resulting histogram
			3,			// it is a 3D histogram
			histSize,	// number of bins
			ranges		// pixel value range
		);

		return hist;
	}

	// Computes the 1D Hue histogram with a mask.
	// BGR source image is converted to HSV
	// Pixels with low saturation are ignored
	cv::MatND getHueHistogram(const cv::Mat &image, 
                             int minSaturation=0) {

		cv::MatND hist;

		// Convert to HSV colour space
		cv::Mat hsv;
		cv::cvtColor(image, hsv, CV_BGR2HSV);

		// Mask to be used (or not)
		cv::Mat mask;

		if (minSaturation>0) {
		
			// Spliting the 3 channels into 3 images
			std::vector<cv::Mat> v;
			cv::split(hsv,v);

			// Mask out the low saturated pixels
			cv::threshold(v[1],mask,minSaturation,255,
                                 cv::THRESH_BINARY);
		}

		// Prepare arguments for a 1D hue histogram
		hranges[0]= 0.0;
		hranges[1]= 180.0;
		channels[0]= 0; // the hue channel 

		// Compute histogram
		cv::calcHist(&hsv, 
			1,			// histogram of 1 image only
			channels,	// the channel used
			mask,		// binary mask
			hist,		// the resulting histogram
			1,			// it is a 1D histogram
			histSize,	// number of bins
			ranges		// pixel value range
		);

		return hist;
	}

	// Computes the 2D ab histogram.
	// BGR source image is converted to Lab
	cv::MatND getabHistogram(const cv::Mat &image) {

		cv::Mat hist;

		// Convert to Lab color space
		cv::Mat lab;
		cv::cvtColor(image, lab, CV_BGR2Lab);

		// Prepare arguments for a 2D color histogram
		hranges[0]= -128.0;
		hranges[1]= 127.0;
		channels[0]= 1; // the two channels used are ab 
		channels[1]= 2; 

		// Compute histogram
		cv::calcHist(&lab, 
			1,			// histogram of 1 image only
			channels,	// the channel used
			cv::Mat(),	// no mask is used
			hist,		// the resulting histogram
			2,			// it is a 2D histogram
			histSize,	// number of bins
			ranges		// pixel value range
		);

		return hist;
	}

	/*
	// Computes the 1D Hue histogram with a mask.
	// BGR source image is converted to HSV
	cv::MatND getHueHistogram(const cv::Mat &image) {

		cv::Mat hist;

		// Convert to Lab colour space
		cv::Mat hue;
		cv::cvtColor(image, hue, CV_BGR2HSV);

		// Prepare arguments for a 1D hue histogram
		hranges[0]= 0.0;
		hranges[1]= 180.0;
		channels[0]= 0; // the hue channel 

		// Compute histogram
		cv::calcHist(&hue, 
			1,			// histogram of 1 image only
			channels,	// the channel used
			cv::Mat(),	// no mask is used
			hist,		// the resulting histogram
			1,			// it is a 1D histogram
			histSize,	// number of bins
			ranges		// pixel value range
		);

		return hist;
	}
	*/
	cv::Mat colorReduce(const cv::Mat &image, int div=64) {

	  int n= static_cast<int>(log(static_cast<double>(div))/log(2.0));
	  // mask used to round the pixel value
	  uchar mask= 0xFF<<n; // e.g. for div=16, mask= 0xF0

	  cv::Mat_<cv::Vec3b>::const_iterator it= image.begin<cv::Vec3b>();
	  cv::Mat_<cv::Vec3b>::const_iterator itend= image.end<cv::Vec3b>();

	  // Set output image (always 1-channel)
	  cv::Mat result(image.rows,image.cols,image.type());
	  cv::Mat_<cv::Vec3b>::iterator itr= result.begin<cv::Vec3b>();

	  for ( ; it!= itend; ++it, ++itr) {
        
        (*itr)[0]= ((*it)[0]&mask) + div/2;
        (*itr)[1]= ((*it)[1]&mask) + div/2;
        (*itr)[2]= ((*it)[2]&mask) + div/2;
	  }

	  return result;
}

};


#endif
