#include "utils.h"
#include <sstream>
using std::stringstream;
#include <iostream>
using std::cout;
using std::endl;
#include <algorithm>
using std::min;
using std::max;
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip> 
using std::string;
using std::vector;


cv::Scalar getRandcolor(){
    //std::srand((unsigned)time(0)); 
    int i = (std::rand()%256); 
    int j = (std::rand()%256); 
    int k = (std::rand()%256); 
    return cv::Scalar(i,j,k);
}

vector<cv::Rect2d> getHumanObjects(vector<vector<float>> detections, float confidenceThreshold, cv::Mat img){
    vector<cv::Rect2d> objects;
    for (int i = 0; i < detections.size(); ++i) {
        const vector<float>& d = detections[i];
        // Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
        //CHECK_EQ(d.size(), 7);
        const float score = d[2];
        if (score >= confidenceThreshold) {
            if (int(d[1])== 1 ) {
                int xmin = std::max( int( d[3] * img.cols ), 0 );
                int ymin = std::max( int( d[4] * img.rows ), 0 );
                int xmax = std::min( int( d[5] * img.cols ), img.cols );
                int ymax = std::min( int( d[6] * img.rows ), img.rows );
                int w = xmax-xmin;
                int h = ymax-ymin;
                float hwr=h/float(w);
                if ( hwr < 6.0 && hwr > 1.2 )
                    objects.push_back( cv::Rect2d( xmin, ymin, w, h ) );
                //return 0;
            }
        }
    } // end for-detections
    return objects;
}

