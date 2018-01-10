#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include "detector.h"
#include "utils.h"
#include "perspective.h"

#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <utility>
#include <vector>
using std::vector;
#include <sstream>
#include <string>
#include <iostream>
#include <fstream>
using std::cout;
using std::endl;
using std::string;



int main(int argc, char* argv[])
{

    cv::Mat img = cv::imread("test.jpg"); 

    //detection config
    const string detectorModel = "models/deploy.prototxt";
    const string detectorWeights = "models/person0712_iter_180000.caffemodel";
    const string detectorMeanFile = "";
    const string detectorMeanString = "104,117,123";
    const float confidenceThreshold = 0.3;
    Detector detector( detectorModel, detectorWeights, detectorMeanFile, detectorMeanString);

    // init perspective    
    string ptsFilename("pointsForPersp.txt");
    Perspective perspective(ptsFilename);
    
    //get warped img
    cv::Mat warped_img = perspective.WarpImg(img, true);

    //detect in warped_img
    vector<vector<float> > detections = detector.Detect(warped_img);
    vector<cv::Rect2d> objects = getHumanObjects(detections, confidenceThreshold,warped_img);

    cv::Mat imageShow = warped_img.clone(); //show detect in warped_img
    int detNum = objects.size();
    for ( int i = 0 ; i < detNum ; ++i ) {
        cv::rectangle( imageShow, 
            cv::Point( objects[i].x, objects[i].y ),
            cv::Point( objects[i].x+objects[i].width, objects[i].y+objects[i].height ),
            cv::Scalar(255,0,0), 2 );
    }
    cv::imshow( "detect_warped_img", imageShow );

    //get trapezoid points in ori img, from detection bbox on warped_img
    vector<vector<cv::Point2f>> pstrapezs;
    for ( int i = 0 ; i < detNum ; ++i ) {
        vector<cv::Point2f> pstrapez=perspective.GetTrapezoidPtsFromRect(objects[i]);
        pstrapezs.push_back(pstrapez);
    }

    //show 3d drawing in img before warp
    imageShow = img.clone();
    for ( int i = 0 ; i < detNum ; ++i ) {
        cv::Scalar randcolor = getRandcolor();
        perspective.Draw3DBox(imageShow,pstrapezs[i],randcolor,2);
    }
    cv::imshow( "3D_detect_on_img_before_warp", imageShow );
    cv::waitKey(0);

    return 0;
}
