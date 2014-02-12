#include "stdafx.h"
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>
#include <list>
#include <cmath>
#include <iomanip>

using namespace std;
using namespace cv;


class PixelData
{
	double weight;
	double mu;
	double var;
	int recentObservation;

public:
	PixelData(int observation, double var, double weight) {
		mu = recentObservation = observation;
		this->var = var;
		this->weight = weight;
	}

	double getWeight() { return weight; }
	double getMu() { return mu; }
	double getVar() { return var; }
	uchar getRecentObservation() { return recentObservation; }

	void setWeight(double weight) { this->weight = weight; }
	void setMu(double mu) { this->mu = mu; }
	void setVar(double var) { this->var = var; }
	void setRecentObservation(uchar recentObservation) { this->recentObservation = recentObservation; }
	bool operator<(const PixelData& pv) { return  (weight / sqrt(var)) > (pv.weight / sqrt(pv.var)); }
};

class PixelBackgroundModel
{
	const double PARAM_ALPHA = 0.1;
	const double PARAM_SIGMA = 2.5;
	const double PARAM_STH = 20;
	const double PARAM_N = 3;
	const double PARAM_INIT_WEIGHT = 0.001;
	const double PARAM_INIT_SIGMA = 30;

	list<PixelData> pixelData;

	double getSumOfWeights() {
		double sum = 0;
		list<PixelData>::iterator iter = pixelData.begin();

		while (iter != pixelData.end()) {
			sum += iter->getWeight();
			iter++;
		}

		return sum;
	}

	void normalizeWeights() {
		double sumOfWeights = getSumOfWeights();

		list<PixelData>::iterator iter = pixelData.begin();

		while (iter != pixelData.end()) {
			iter->setWeight(iter->getWeight() / sumOfWeights);
			iter++;
		}
	}

	void dropLeastProbable() {
		pixelData.sort();
		pixelData.pop_back();
	}

public:
	void updateModel(uchar pixelValue) {
		bool matchFound = false;
		
		list<PixelData>::iterator iter = pixelData.begin();

		while (iter != pixelData.end()) {
			if (!matchFound && abs(pixelValue - iter->getMu()) <= sqrt(iter->getVar()) * PARAM_SIGMA) {
				matchFound = true;
				iter->setRecentObservation(pixelValue);
				iter->setMu((1 - PARAM_ALPHA) * iter->getMu() + PARAM_ALPHA * pixelValue);
				iter->setVar((1 - PARAM_ALPHA) * iter->getVar() + PARAM_ALPHA * (pixelValue - iter->getMu()) * (pixelValue - iter->getMu()));
				iter->setWeight((1 - PARAM_ALPHA) * iter->getWeight() + PARAM_ALPHA);
			} else {
				iter->setWeight((1 - PARAM_ALPHA) * iter->getWeight());
			}
			iter++;
		}


		if (!matchFound) {
			if (pixelData.size() == PARAM_N) {
				dropLeastProbable();
			}
			pixelData.push_back(PixelData(pixelValue, PARAM_INIT_SIGMA * PARAM_INIT_SIGMA, PARAM_INIT_WEIGHT));
		}

		normalizeWeights();
		pixelData.sort();
	}

	uchar getMask(uchar pixelValue) {
		return abs(pixelData.front().getRecentObservation() - pixelValue) <= PARAM_STH ? 0 : 255;
	}

};

class SceneBackgroundModel
{
	PixelBackgroundModel** pixelBackgroundModel;
	Mat mask;
public:
	SceneBackgroundModel(){}
	SceneBackgroundModel(int rows, int cols) {
		pixelBackgroundModel = new PixelBackgroundModel*[rows];
		for (int i = 0; i < rows; i++) {
			pixelBackgroundModel[i] = new PixelBackgroundModel[cols];
		}

		mask = Mat(rows, cols, CV_8U);
	}

	void updateModel(Mat frame) {
		for (int i = 0; i < frame.size().height; i++) {
			for (int j = 0; j < frame.size().width; j++) {
				pixelBackgroundModel[i][j].updateModel(frame.at<uchar>(i, j));
				mask.at<uchar>(i, j) = pixelBackgroundModel[i][j].getMask(frame.at<uchar>(i, j));
			}
		}
	}

	Mat getForegroundMask() {
		return mask;
	}
};

class ForegroundDetector 
{
	SceneBackgroundModel sceneBackgroundModel;
public:
	ForegroundDetector(){}

	ForegroundDetector(int rows, int cols) {
		sceneBackgroundModel = SceneBackgroundModel(rows, cols);
	}

	void setInput(Mat frame) {
		sceneBackgroundModel.updateModel(frame);
	}

	Mat getOutput() {
		return sceneBackgroundModel.getForegroundMask();
	}
};

void processTestSequence(string dirInputPath, string baseFilename, int startFrame, int endFrame, int fixedWidth, string inputFormat, bool outputToFile, string dirOutputPath, string outputFormat, bool showPreview, int previewDelay)
{
	Mat inputImage;
	Mat inputGrayImage;
	Mat outputImage;

	ForegroundDetector* foregroundDetector = NULL;

	for (int i = startFrame; i <= endFrame; i++)
	{
		ostringstream inputFilename, outputFilename;

		if (fixedWidth == -1) {
			inputFilename << dirInputPath << baseFilename << i << "." << inputFormat;
			if (outputToFile) outputFilename << dirOutputPath << baseFilename << i << "." << outputFormat;
		} 
		else {
			inputFilename << dirInputPath << baseFilename << setw(fixedWidth) << setfill('0') << i << "." << inputFormat;
			if (outputToFile) outputFilename << dirOutputPath << baseFilename << setw(fixedWidth) << setfill('0') << i << "." << outputFormat;
		}

		inputImage = imread(inputFilename.str(), CV_LOAD_IMAGE_COLOR);

		cvtColor(inputImage, inputGrayImage, CV_RGB2GRAY);

		if (!inputImage.data) {
			cout << "Could not open or find the image: " << std::endl << inputFilename.str() << std::endl;
			return;
		}

		if (i == startFrame) {
			foregroundDetector = new ForegroundDetector(inputGrayImage.size().height, inputGrayImage.size().width);
		} 
		
		foregroundDetector->setInput(inputGrayImage);
		outputImage = foregroundDetector->getOutput();

		if (showPreview) {
			namedWindow("Input", CV_WINDOW_AUTOSIZE);
			imshow("Input", inputImage);

			namedWindow("Output", CV_WINDOW_AUTOSIZE);
			imshow("Output", outputImage);
		}
		
		waitKey(previewDelay);

		if (outputToFile) imwrite(outputFilename.str(), outputImage);

	}
}

int main(int argc, char** argv)
{
	string inputDir = "C:\\Shuvo\\Research\\PhD\\PhD-Datasets\\TestSequence\\PETS\\PETS2000\\";
	string baseFilename = "image";
	int startFrame = 1; 
	int endFrame = 500; 
	int fixedWidth = -1;
	string inputFormat = "jpg";
	bool showPreview = true;
	bool outputToFile = false;
	string outputDir = "C:\\Shuvo\\Development\\Projects\\vision-foreground-detector\\Output\\";
	string outputFormat = "jpg";
	int previewDelay = 10;

	processTestSequence(inputDir, baseFilename, startFrame, endFrame, fixedWidth, inputFormat, outputToFile, outputDir, outputFormat, showPreview, previewDelay);

	return 0;
}



