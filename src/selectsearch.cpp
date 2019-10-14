// References used: Copyright (C) 2018-2019, BigVision LLC (LearnOpenCV.com), All Rights Reserved.
// Mask R-CNN object detection algorithm
// Developed further by:
// Tlotliso Mapana
// MPNTLO002

#include <opencv2/ximgproc/segmentation.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <ctime>

using namespace std;
using namespace cv;
using namespace cv::ximgproc::segmentation;

int main (int argc, char** argv){
	clock_t start = clock();
	double duration;
	
	// read image
	Mat image = cv::imread("data/ship.jpg");
	// resize image
	
	int newHeight = 300;
	
	int newWidth = round((image.cols*newHeight)/image.rows);
		
	cv::resize(image, image, Size(newWidth, newHeight));
	
	// create selective search segmentation object
	Ptr<SelectiveSearchSegmentation> ss = createSelectiveSearchSegmentation();
	
	// set input image on which to run segmentation and run
	ss->setBaseImage(image);

	// Try fast selective search
	ss->switchToSelectiveSearchFast();

	// run selective search algorithm
	vector<Rect> rects;
	ss->process(rects);
	cout << "Possible number of regions: " << rects.size() << endl << endl;

	//int numShowRects = 10;	// control number of boxes that can be drawn
	int numShowRects = rects.size();

	Mat imageOut = image.clone();
	while(1){
		for(int i=0; i<(int)rects.size(); i++){
			if(i < numShowRects)
				rectangle(imageOut, rects[i], Scalar(0, 255, 0));
			else
				break;
		}

		break;
	}
	
	duration = (clock() - start) / (double) CLOCKS_PER_SEC;
	cout << "Time to complete: " << duration << endl << endl;
	
	cout << numShowRects << " rectangles" << endl;
	
	// output result
	imwrite("data/SSOut.jpg", imageOut);

	return 0;
}
