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

#include "integral.h"

int main()
{
	cv::Mat image= cv::imread("book.jpg",0);
	if (!image.data)
		return 0; 

	cv::namedWindow("image");
	cv::imshow("image",image);

	cv::Mat binaryFixed;
	cv::Mat binaryAdaptive;
	cv::threshold(image,binaryFixed,70,255,cv::THRESH_BINARY);

	int64 time;
	time= cv::getTickCount();
	cv::adaptiveThreshold(image,binaryAdaptive,255,cv::ADAPTIVE_THRESH_MEAN_C,cv::THRESH_BINARY,21,9);
	time= cv::getTickCount()-time;
	std::cout << "time (adaptiveThreshold)= " << time << std::endl; 

	IntegralImage<int,1> integral(image);

	std::cout << "sum=" << integral(18,45,30,50) << std::endl;

	cv::Mat test(image,cv::Rect(18,45,30,50));
	cv::Scalar t= cv::sum(test);
	std::cout << "sum test=" << t[0] << std::endl;

	std::cout << "sum=" << integral(65,115,15) << std::endl;
	cv::Mat test2(image,cv::Rect(50,100,31,31));	
	t= cv::sum(test2);
	std::cout << "sum test=" << t[0] << std::endl;

	cv::namedWindow("Fixed Threshold");
	cv::imshow("Fixed Threshold",binaryFixed);

	cv::namedWindow("Adaptive Threshold");
	cv::imshow("Adaptive Threshold",binaryAdaptive);

	time= cv::getTickCount();
	  cv::Mat binary= image.clone();
	  int nl= binary.rows; // number of lines
	  int nc= binary.cols; // total number of elements per line
              
	  cv::Mat iimage;
	  cv::integral(image,iimage,CV_32S);

      for (int j=10; j<nl-10; j++) {

		  // get the address of row j
		  uchar* data= binary.ptr<uchar>(j);
		  int* idata1= iimage.ptr<int>(j-10);
		  int* idata2= iimage.ptr<int>(j+10);

          for (int i=10; i<nc-10; i++) {
 
            // process each pixel ---------------------

//			  if (data[i]<integral(i,j,10)/(21*21) - 9)
			  int sum= (idata2[i+10]-idata2[i-10]-idata1[i+10]+idata1[i-10])/(21*21);
			  if (data[i]<sum - 9)
				  data[i]= 0;
			  else
				  data[i]=255;
 
            // end of pixel processing ----------------
 
          } // end of line                   
      }
	time= cv::getTickCount()-time;
	std::cout << "time integral= " << time << std::endl; 

	cv::namedWindow("Adaptive Threshold (integral)");
	cv::imshow("Adaptive Threshold (integral)",binary);

	time= cv::getTickCount();
	cv::Mat filtered;
	cv::Mat binaryFiltered;
	cv::boxFilter(image,filtered,CV_8U,cv::Size(21,21));
	filtered= filtered-9;
	binaryFiltered= image>= filtered;
	time= cv::getTickCount()-time;

	std::cout << "time filtered= " << time << std::endl; 

	cv::namedWindow("Adaptive Threshold (filtered)");
	cv::imshow("Adaptive Threshold (filtered)",binaryFiltered);

	cv::waitKey();
}
