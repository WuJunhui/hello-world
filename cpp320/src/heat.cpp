#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/tracking.hpp>
#include "detector.h"
//#include "tracker.h"
#include "utils.h"

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

#include <chrono>
typedef std::chrono::high_resolution_clock Clock;



int main(int argc, char* argv[])
{
    if(argc!=5){
        cout<<"usage:./test videofile matrixfile detectconfigfile outputtxtfile"<<endl;
        return 1;
    }

    //get video file name
    std::string videofile(argv[1]);
    std::string matrixfile(argv[2]);
    std::string configfile(argv[3]);
    std::string txtfile(argv[4]);
    cout<<"videofile name: "<<videofile<<endl;
    cout<<"matrixfile name: "<<matrixfile<<endl;
    cout<<"detectconfigfile name: "<<configfile<<endl;
    cout<<"output name: "<<txtfile<<endl;
    // initialization caffemodel
    // read config from file
    std::ifstream infile(configfile);
    if(!infile) {
        cout << "Cannot open detection config file.\n";
        return 1;
    }

    string detectorModel;
    string detectorWeights;
    string detectorMeanString;
    string detectIntervalString;
    string confidenceThresholdString;
    string LONG_EDGE_LIMITString;
    std::getline(infile, detectorModel);
    std::getline(infile, detectorWeights);
    std::getline(infile, detectorMeanString);
    std::getline(infile, detectIntervalString);
    std::getline(infile, confidenceThresholdString);
    std::getline(infile, LONG_EDGE_LIMITString);
    infile.close();

    //const string detectorModel = "models/deploy.prototxt";
    //const string detectorWeights = "models/person0712_iter_180000.caffemodel";
    const string detectorMeanFile = "";
    //const string detectorMeanString = "104,117,123";

    const int detectInterval = atoi(detectIntervalString.c_str());
    const float confidenceThreshold = strtof(confidenceThresholdString.c_str(),0);

    auto t00 = Clock::now();
    Detector detector( detectorModel, detectorWeights, detectorMeanFile, detectorMeanString);
    auto t01 = Clock::now();
    cout<<"Ini detection time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(t01 - t00).count()<<endl;

    //init about perspective
    const int LONG_EDGE_LIMIT=640;
    float longEdge;
    float resize_scale=1.0;
    cv::Mat m_perspective= readMatFromTxt(matrixfile,3,3);

#ifdef MY_VIS_MODE //init top-view img
    const int PROJ_W=1191;
    const int PROJ_H=1813;
    cv::Mat roi_image;
    vector<cv::Scalar> colors;
    cv::Scalar randcolor;
#endif

    //tracking config
    string trackingAlg = "KCF";
    vector<cv::MultiTracker> trackersvector;
    //vector<cv::Ptr<cv::Tracker>> algorithms;
    vector<vector<cv::Point>> traces;
    
    //txtout
    std::ofstream txtout(txtfile);


    //open video
    auto t02 = Clock::now();
    cv::VideoCapture cap(videofile);
    if (!cap.isOpened()) {
      LOG(FATAL) << "Failed to open video: " << videofile;
    }
    auto t03 = Clock::now();
    cout<<"openvideo time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(t03 - t02).count()<<endl;

    int frame_count = 0;
    cv::Mat img;

    while (true) {
        //cout<<"Processing frame_count "<<frame_count<<"...."<<endl;
        auto t0 = Clock::now();
        bool success = cap.read(img);
        if (!success) {
          LOG(INFO) << "Process " << frame_count << " frames from " << videofile;
          break;
        }
        CHECK(!img.empty()) << "Error when read frame";
        
        //init original img;
        if (frame_count ==0){
            cv::Mat original_img=img.clone();
            longEdge = float(std::max(original_img.cols,original_img.rows));
            resize_scale=LONG_EDGE_LIMIT/longEdge;
#ifdef MY_VIS_MODE //get top-view img
            cv::warpPerspective(original_img,roi_image,m_perspective,cv::Size(PROJ_W,PROJ_H));
#endif
        }

        //resize every frame
        cv::resize(img,img,cv::Size(),resize_scale,resize_scale);

        // for each frame in detectInterval: detect the persons
        if ( frame_count % detectInterval == 0 ) {
            if(frame_count != 0){
                writeTrace(traces,txtout,frame_count);
#ifdef MY_VIS_MODE //vis traces in top-view img
                for (int i=0; i<traces.size();i++){
                    randcolor=getRandcolor();
                    for (int j=1; j<traces[i].size();j++)
                        cv::circle( roi_image, traces[i][j], 10,randcolor, 6);
                }
                cv::Mat rect_img=roi_image.clone();
                cv::resize(rect_img,rect_img,cv::Size(),0.3,0.3);
                cv::imshow("trace",rect_img);
                cv::waitKey(10);
                if (cv::waitKey(1) == 27)
                    return 0;       
#endif

            }            
            auto t1 = Clock::now();
            //start detect
            vector<vector<float> > detections = detector.Detect(img);
            vector<cv::Rect2d> objects = getHumanObjects(detections, confidenceThreshold,img);//objects from det
            //vector<cv::Rect2d> objects; 

            int detNum = objects.size();
            auto t2 = Clock::now();
#ifdef MY_VIS_MODE // vis detection result
            cv::Mat imageShow = img.clone(); //for show
            char buf[10];
            sprintf( buf, "%d", frame_count );
            cv::putText( imageShow, string( buf ), cv::Point(30,30), 
                    cv::FONT_HERSHEY_TRIPLEX, 0.8, cv::Scalar(0,0,255), 1, false );

            for ( int i = 0 ; i < detNum ; ++i ) {
                cv::rectangle( imageShow, 
                    cv::Point( objects[i].x, objects[i].y ),
                    cv::Point( objects[i].x+objects[i].width, objects[i].y+objects[i].height ),
                    cv::Scalar(255,0,0), 2 );
            }
            cv::imshow( "detect", imageShow );
            cv::waitKey(100);
            if (cv::waitKey(1) == 27)
                return 0;  
#endif
            auto t5 = Clock::now();
            cout<<"read img time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(t1- t0).count()<<endl;
            cout<<"Detection time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()<<endl;
            cout<<"Display time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t2).count()<<endl;
            
            //clear multitrackers
            if(frame_count !=0){
                //trackersvector[0].~MultiTracker();
                trackersvector.clear();
                //algorithms.clear();
                traces.clear();
            }
            cv::MultiTracker trackers(trackingAlg);
            trackersvector.push_back(trackers);

            //create trackers and add tracking objects 
            for (int i=0; i < detNum ; ++i){
                vector<cv::Point> tracepoints;
                //algorithms.push_back(createTrackerByName(trackingAlg));
                traces.push_back(tracepoints);//reserve traces
            }
            //trackersvector[0].add(algorithms,img,objects);
            trackersvector[0].add(img,objects);
        }

        //update trackers in each frame

        auto ttrack0 = Clock::now();
        trackersvector[0].update(img);
        auto ttrack1 = Clock::now();
        cout<<"frame "<<frame_count<<" track time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(ttrack1- ttrack0).count()<<endl;
        //vector<cv::Rect2d> trackobjs(trackersvector[0].getObjects());
        vector<cv::Rect2d> trackobjs=trackersvector[0].objects;

#ifdef MY_VIS_MODE //vis trackobjs in original img
        cv::Mat imageShow = img.clone(); //for show
        for(int i=0;i<trackobjs.size();i++){
            cv::rectangle( imageShow, trackobjs[i], cv::Scalar( 0, 0, 255 ), 2, 1 );
        }
        cv::imshow("tracker",imageShow);
        cv::waitKey(10);
        if (cv::waitKey(1) == 27)
            return 0;    
 
#endif

        vector<cv::Point> foot_pts = getFoot(trackobjs, m_perspective,resize_scale);
        for(int i=0;i<foot_pts.size();i++){
            traces[i].push_back(foot_pts[i]);
        }

        //count add 
        ++frame_count;
    }
    if (cap.isOpened()) 
        cap.release();
    txtout.close();
    return 0;
}


