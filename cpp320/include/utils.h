
#include <map>
using std::map;
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
#include <opencv2/tracking.hpp>
#include <algorithm>

void prepareClassesMap( const string &class_file, std::map<int, string> &classMap );

float distanceBoxes( const cv::Rect &bbox1, const cv::Rect &bbox2 );

float IOU( const cv::Rect &bbox1, const cv::Rect &bbox2 );

void l2_normalize( const vector<float> &src, vector<float> &dest );

float l2_dis( const vector<float> &vec1, const vector<float> &vec2 );


cv::Mat readMatFromTxt(string filename, int rows,int cols);

vector<cv::Point> getFoot(vector<cv::Rect2d> bboxes, cv::Mat M, float scale);

//cv::Ptr<cv::Tracker> createTrackerByName(cv::String name);

void writeTrace(vector<vector<cv::Point>> traces, std::ostream& out, int frame_count);

cv::Scalar getRandcolor();

vector<cv::Rect2d> getHumanObjects(vector<vector<float>> detections, float confidenceThreshold, cv::Mat img);

