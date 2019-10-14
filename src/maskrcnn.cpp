// References used: Copyright (C) 2018-2019, BigVision LLC (LearnOpenCV.com), All Rights Reserved.
// Mask R-CNN object detection algorithm
// Developed further by:
// Tlotliso Mapana
// MPNTLO002

#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <string>

using namespace cv;
using namespace dnn;
using namespace std;

// Threshold parameters for the binary mask
float confidenceThresh = 0.5;
float maskThresh = 0.4;

vector<string> classes;
vector<Scalar> colours;

// draw predicted bounding box with a chosen colour
void drawBoundingBox(Mat& frame, int classID, float confidence, Rect boundingBox, Mat& mask){
	// draw rectangle
	rectangle(frame, Point(boundingBox.x, boundingBox.y), Point(boundingBox.x+boundingBox.width, boundingBox.y+boundingBox.height), Scalar(255, 150, 50), 3);
	
	// extract classID and label
	string label = format("%.2f", confidence);
	if(!classes.empty()){
		CV_Assert(classID < (int)classes.size());
		if(classID < (int)classes.size())
			label = classes[classID] + ": " + label;
	}
	
	// get name and put text on top on bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	boundingBox.y = max(boundingBox.y, labelSize.height);
	rectangle(frame, Point(boundingBox.x, boundingBox.y), Point(boundingBox.x+round(1.5*labelSize.width), boundingBox.y+baseLine), Scalar(255, 255, 255), FILLED);
	putText(frame, label, Point(boundingBox.x, boundingBox.y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0), 1);
	
	Scalar colour = colours[classID % colours.size()];
	
	// apply mask of the image
	resize(mask, mask, Size(boundingBox.width, boundingBox.height));
	Mat imgMask = (mask > maskThresh);
	Mat colouredROI = (0.5*colour + 0.5 * frame(boundingBox));
	colouredROI.convertTo(colouredROI, CV_8UC3);
	
	// extra work
	vector<Mat> contours;
	Mat hierarchy;
	imgMask.convertTo(imgMask, CV_8U);
	findContours(imgMask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
	drawContours(colouredROI, contours, -1, colour, 5, LINE_8, hierarchy, 100);
	colouredROI.copyTo(frame(boundingBox), imgMask);
}

// to extract a bounding box and a binary mask for each detection 'object'
void postprocess(Mat& frame, const vector<Mat>& outputs){
	Mat detections, masks;
	
	detections = outputs[0];
	masks = outputs[1];
	
	const int numDetections = detections.size[2];
	
	detections = detections.reshape(1, detections.total()/7);
	
	for(int i=0; i<numDetections; ++i){
		float score = detections.at<float>(i, 2);
		if(score > confidenceThresh){
			// extract bounding box
			int classID = static_cast<int>(detections.at<float>(i, 1));
			int x_1 = static_cast<int>(frame.cols * detections.at<float>(i, 3));
			int y_1 = static_cast<int>(frame.rows * detections.at<float>(i, 4));
			int x_2 = static_cast<int>(frame.cols * detections.at<float>(i, 5));
			int y_2 = static_cast<int>(frame.rows * detections.at<float>(i, 6));
			
			// create the box bounded by [x_1,y_1] and [x_2,y_2]
			x_1 = max(0, min(x_1, frame.cols-1));
			y_1 = max(0, min(y_1, frame.rows-1));
			x_2 = max(0, min(x_2, frame.cols-1));
			y_2 = max(0, min(y_2, frame.rows-1));
			Rect boundingBox = Rect(x_1, y_1, x_2-x_1+1, y_2-y_1+1);
			
			// extract the binary mask by the classID
			Mat mask(masks.size[2], masks.size[3], CV_32F, masks.ptr<float>(i, classID));
			
			// call the drawBoundingBox function
			drawBoundingBox(frame, classID, score, boundingBox, mask);
		}
	}
}

int main(int argc, char **argv){
	clock_t start = clock();
	double duration;
	
	// Load the class names
	string classesFile = "data/labels.names ";
	ifstream inClassFile(classesFile.c_str());
	
	string line;
	while(getline(inClassFile, line)){
		classes.push_back(line);
	}
	
	// Load the colours used to mask the objects
	string coloursFile = "data/colours.txt";
	ifstream inColoursFile(coloursFile.c_str());
	
	while(getline(inColoursFile, line)){
		char* pEnd;
		double r, g, b;
		r = strtod(line.c_str(), &pEnd);
		g = strtod(pEnd, NULL);
		b = strtod(pEnd, NULL);
		colours.push_back(Scalar(r, g, b, 255.0));
	}
	
	// Extract the model configurations
	string textGraph = "data/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
	string modelWeights = "data/mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb";
	
	// Load the network
	Net net = readNetFromTensorflow(modelWeights, textGraph);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);
	
	cout << "Network created successfully!" << endl << endl;
	
	// Prepare output video file
	string videoFile, outputFile;
	VideoCapture cap;
	VideoWriter video;
	Mat frame, blob;
	
	videoFile = "data/video_02_preview.avi";
	
	try{
		cap.open(videoFile);
		//cap.open(0);	// uncomment for camera
		cout << "Video stream opened" << endl << endl;
	}
	catch(runtime_error& re){
		cout << "Unable to open video stream!" << endl;
		return 0;
	}
	
	// initialise video writer to save output
	//outputFile = "data/mask_rnn_output.avi";
	//video.open(outputFile, VideoWriter::fourcc('M', 'J', 'P', 'G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
	// will save individual frames instead using imsave())
	
	int count = 0;
	
	while(waitKey(1) < 0){
		// get a frame from video stream
		cap >> frame;
		
		//if(frame.empty()){	// Reached end of video
		if(count > 9){	// only 10 frames for testing
			cout << "End of video" << endl << endl;
			waitKey(3000);
			break;
		}
		
		video.write(frame);
		
		// create blob from a frame
		blobFromImage(frame, blob); // uncomment
		
		// set the input to the network created earlier
		net.setInput(blob);
		
		// outputs
		vector<string> outputNames(2);
		outputNames[0] = "detection_out_final";
		outputNames[1] = "detection_masks";
		vector<Mat> outputs;
		
		net.forward(outputs, outputNames);
		postprocess(frame, outputs);
		
		// write frame with detections
		Mat detectedFrame;
		frame.convertTo(detectedFrame, CV_8U);
		string frameName = "data/"+to_string(count)+".jpg";
		imwrite(frameName, detectedFrame);
		
		count++;
				
		if((char)waitKey(1) == 27)
			break;
	}
	
	duration = (clock() - start) / (double) CLOCKS_PER_SEC;
	cout << "Time to complete: " << duration << endl << endl;
	
	cap.release();
	video.release();
	
	return 0;
}
