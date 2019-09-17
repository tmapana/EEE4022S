// This is a selective search algorithm by Tlotliso Mapana
// Code has been adopted from www.LearnOpenCV.com

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
	// read image
	Mat image = cv::imread("../data/ship.jpg");
	// resize image
	int newHeight = 200;
	int newWidth = (image.cols*newHeight)/image.rows;
	cout << "New Width: " << newWidth << endl;
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
	cout << "Possible number of regions: " << rects.size() << endl;

	int numShowRects = 100;
	int increment = 50;

	while(1){
		Mat imageOut = image.clone();

		for(int i=0; i<rects.size(); i++){
			if(i < numShowRects)
				rectangle(imageOut, rects[i], Scalar(0, 255, 0));
			else
				break;
		}

		cout << numShowRects << " iterations done." << endl;
		// show output
		//imshow("Output", imageOut);

		break;
	}

	return 0;
}
