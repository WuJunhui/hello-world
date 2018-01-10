#include <vector>
using std::vector;
#include <string>
using std::string;
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip> 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>


cv::Scalar getRandcolor();

vector<cv::Rect2d> getHumanObjects(vector<vector<float>> detections, float confidenceThreshold, cv::Mat img);
