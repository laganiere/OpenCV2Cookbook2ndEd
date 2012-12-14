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

#if !defined COLORDETECT
#define COLORDETECT

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class ColorDetector {

  private:

	  // minimum acceptable distance
	  int minDist; 

	  // target color
	  cv::Vec3b target; 

	  // image containing resulting binary map
	  cv::Mat result;

  public:

	  // empty constructor
	  ColorDetector() : minDist(100) { 

		  // default parameter initialization here
		  target[0]= target[1]= target[2]= 0;
	  }

	  // other constructor
	  ColorDetector(uchar blue, uchar green, uchar red, int minDist=100): minDist(minDist) { 

		  // target color
		  setTargetColor(blue, green, red);
	  }

	  // Computes the distance from target color.
	  int getDistanceToTargetColor(const cv::Vec3b& color) const {
		  return getColorDistance(color, target);
	  }

	  // Computes the city-block distance between two colors.
	  int getColorDistance(const cv::Vec3b& color1, const cv::Vec3b& color2) const {

		  return abs(color1[0]-color2[0])+
					abs(color1[1]-color2[1])+
					abs(color1[2]-color2[2]);

	 	  // Or:
		  // return static_cast<int>(cv::norm<int,3>(cv::Vec3i(color[0]-color2[0],color[1]-color2[1],color[2]-color2[2])));
		  
		  // Or:
		  // cv::Vec3b dist;
		  // cv::absdiff(color,color2,dist);
		  // return cv::sum(dist)[0];
	  }

	  // Processes the image. Returns a 1-channel binary image.
	  cv::Mat process(const cv::Mat &image);

	  cv::Mat operator()(const cv::Mat &image) {
	  
		  cv::Mat output;
		  // compute absolute difference with target color
		  cv::absdiff(image,cv::Scalar(target),output);
	      // split the channels into 3 images
	      std::vector<cv::Mat> images;
	      cv::split(output,images);
		  // add the 3 channels
	      output= images[0]+images[1]+images[2];
		  // apply threshold
          cv::threshold(output,  // input image
                      output,  // output image
                      minDist, // threshold
                      255,     // max value
                 cv::THRESH_BINARY_INV); // thresholding type
	
	      return output;
	  }

	  // Getters and setters

	  // Sets the color distance threshold.
	  // Threshold must be positive, otherwise distance threshold
	  // is set to 0.
	  void setColorDistanceThreshold(int distance) {

		  if (distance<0)
			  distance=0;
		  minDist= distance;
	  }

	  // Gets the color distance threshold
	  int getColorDistanceThreshold() const {

		  return minDist;
	  }

	  // Sets the color to be detected
	  void setTargetColor(uchar blue, uchar green, uchar red) {

		  target[2]= red;
		  target[1]= green;
		  target[0]= blue;
	  }

	  // Sets the color to be detected
	  void setTargetColor(cv::Vec3b color) {

		  target= color;
	  }

	  // Gets the color to be detected
	  cv::Vec3b getTargetColor() const {

		  return target;
	  }
};


#endif
