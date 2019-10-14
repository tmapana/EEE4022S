// References used: Copyright (C) 2018-2019, BigVision LLC (LearnOpenCV.com), All Rights Reserved.
// Mask R-CNN object detection algorithm
// Developed further by:
// Tlotliso Mapana
// MPNTLO002

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <stdio.h>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

Mat img, grayImg, cln, edges;

int main(int argc, char* argv[]){
	clock_t start = clock();
	double duration;
	
	// read image as RGB and convert to grayscale
	img = imread("data/ship.jpg", IMREAD_COLOR);
	if(img.empty()){
		cout << "Image not found" << endl << endl;
		return -1;
	}

	cvtColor(img, grayImg, COLOR_BGR2GRAY);

	// clone img
	cln = img.clone();

	// store detected edges
	Canny(img, edges, 150, 400);

	// create vector to store lines of the image
	vector<Vec4i> lines;

	// apply Hough Transform
	HoughLinesP(edges, lines, 1, CV_PI/180, 150, 100, 50);

	// iteratively draw lines on the image
	for(size_t i=0; i<lines.size(); i++){
		Vec4i l = lines[i];
		line(cln, Point(l[0],l[1]), Point(l[2],l[3]), Scalar(0,0,255), 1, LINE_AA);
	}
	
	duration = (clock() - start) / (double) CLOCKS_PER_SEC;
	cout << "Time to complete: " << duration << endl << endl;
	
	// write output file
	imwrite("data/HTOut.jpg", cln);

	cout << "DONE!" << endl;
	return 0;
}
