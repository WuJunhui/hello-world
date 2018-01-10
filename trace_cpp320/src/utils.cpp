
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
#include <opencv2/tracking.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip> 
using std::string;
using std::vector;

void prepareClassesMap( const string &class_file, std::map<int, string> &classMap )
{
  std::ifstream infile( class_file.c_str() );
  string line;
  while ( getline( infile, line ) ) {
      size_t pos = line.find( ' ' );
      if ( pos == string::npos )
          continue;
      cout<<line<<endl;
      string name = line.substr( 0, pos );
      stringstream convert( line.substr( pos ) );
      int id;
      convert >> id;
      classMap.insert( std::pair<int, string>( id, name ) );
  }
}

float distanceBoxes( const cv::Rect &bbox1, const cv::Rect &bbox2 )
{
    int minx_1 = ( int )( bbox1.x );
    int miny_1 = ( int )( bbox1.y );
    int w_1 = ( int )( bbox1.width );
    int h_1 = ( int )( bbox1.height );
    int minx_2 = ( int )( bbox2.x );
    int miny_2 = ( int )( bbox2.y );
    int w_2 = ( int )( bbox2.width );
    int h_2 = ( int )( bbox2.height );
    float center_1_x = minx_1 + w_1 / 2;
    float center_1_y = miny_1 + h_1 / 2;
    float center_2_x = minx_2 + w_2 / 2;
    float center_2_y = miny_2 + h_2 / 2;
    float tmp1 = center_1_x - center_2_x;
    float tmp2 = center_1_y - center_2_y;
    float dist = sqrt( tmp1 * tmp1 + tmp2 * tmp2 );
    return dist;
}

float IOU( const cv::Rect &bbox1, const cv::Rect &bbox2 )
{
    int minx_1 = ( int )( bbox1.x );
    int miny_1 = ( int )( bbox1.y );
    int w_1 = ( int )( bbox1.width );
    int h_1 = ( int )( bbox1.height );
    int minx_2 = ( int )( bbox2.x );
    int miny_2 = ( int )( bbox2.y );
    int w_2 = ( int )( bbox2.width );
    int h_2 = ( int )( bbox2.height );
    float width = min( ( minx_1 + w_1 ), ( minx_2 + w_2 ) ) - max( minx_1, minx_2 );
    if ( width < 0 )
        return 0.0;
    float height = min( ( miny_1 + h_1 ), ( miny_2 + h_2 ) ) - max( miny_1, miny_2 );
    if ( height < 0 )
        return 0.0;
    float areaInt = width * height;
    float area1 = bbox1.area();
    float area2 = bbox2.area();
    return areaInt / min( area1, area2 );
}

void l2_normalize( const vector<float> &src, vector<float> &dest )
{
    int len = src.size();
    for ( int i = 0 ; i < len ; ++i ) {
        dest.push_back( src[i] );
    }
    float sum = 0.0;
    for ( int i = 0 ; i < len ; ++i ) {
        sum += src[i] * src[i];
    }
    sum = sqrt( sum );
    if ( sum != 0 ) {
        for ( int i = 0 ; i < len ; ++i ) {
            dest[i] /= sum;
        }
    }
}

float l2_dis( const vector<float> &vec1, const vector<float> &vec2 )
{
    int len = vec1.size();
    assert( len == vec2.size() );
    float sum = 0.0;
    for ( int i = 0 ; i < len ; ++i ) {
        float dis = vec1[i] - vec2[i];
        sum += dis * dis;
    }
    return sqrt( sum );
}

cv::Mat readMatFromTxt(string filename, int rows,int cols)
{
    double m;
    cv::Mat out = cv::Mat::zeros(rows, cols, CV_64FC1);//Matrix to store values

    std::ifstream fileStream(filename);
    int cnt = 0;//index starts from 0
    while (fileStream >> m)
    {
        int temprow = cnt / cols;
        int tempcol = cnt % cols;
        //cout<<"temprow,col: "<<temprow<<" "<<tempcol<<endl;
        //cout<<"m: "<<m<<endl;
        out.at<double>(temprow, tempcol) = m;
        cnt++;
    }
    return out;
}

vector<cv::Point> getFoot(vector<cv::Rect2d> bboxes, cv::Mat M, float scale=1.0)
{
    vector <cv::Point> foot_coords;
    int numB=bboxes.size();
    //cout<<"numB: "<<numB<<endl;
    for(int i=0; i<numB; i++){
        float x,y,w,h;
        x=bboxes[i].x;
        y=bboxes[i].y;
        w=bboxes[i].width;
        h=bboxes[i].height;
        float bottom_x=(x+w/2.0)/scale;
        float bottom_y=(y+h)/scale;
        //cout<<"bottom: "<<bottom_x<<" "<<bottom_y<<endl;

        double dB[]={bottom_x,bottom_y,1};
        cv::Mat B = cv::Mat(3,1,CV_64FC1,dB);
        //cout<<"start pro..."<<endl;
        cv::Mat projected = M*B;
        int proj_x=(projected.at<double>(0,0)/projected.at<double>(2,0)+0.5);
        int proj_y=(projected.at<double>(1,0)/projected.at<double>(2,0)+0.5);
        //cout<<"projected: "<<proj_x<<" "<<proj_y<<endl;
        foot_coords.push_back(cv::Point(proj_x,proj_y));
    }
    return foot_coords;
}

/*cv::Ptr<cv::Tracker> createTrackerByName(cv::String name)
{
    cv::Ptr<cv::Tracker> tracker;

    if (name == "KCF")
        tracker = cv::TrackerKCF::create();
    else if (name == "TLD")
        tracker = cv::TrackerTLD::create();
    else if (name == "BOOSTING")
        tracker = cv::TrackerBoosting::create();
    else if (name == "MEDIAN_FLOW")
        tracker = cv::TrackerMedianFlow::create();
    else if (name == "MIL")
        tracker = cv::TrackerMIL::create();
    else if (name == "GOTURN")
        tracker = cv::TrackerGOTURN::create();
    else
        CV_Error(cv::Error::StsBadArg, "Invalid tracking algorithm name\n");

    return tracker;
}*/


void writeTrace(vector<vector<cv::Point>> traces, std::ostream& out, int frame_count){
    stringstream ff;
    ff<<std::setfill('0')<<std::setw(9)<<frame_count;
    out<<"FRAME: "<<ff.str()<<endl;
    for (int i=0; i<traces.size(); i++){
        out<<"[";
        vector<cv::Point> tracepoints=traces[i];
        for (int j=0; j<tracepoints.size();j++){
            if (j!=0)
                out<<", ";
            out<<"("<<tracepoints[j].x<<", "<<tracepoints[j].y<<")";
        }

        out<<"]"<<endl;
    }
}

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

