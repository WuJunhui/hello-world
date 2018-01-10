#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

using std::vector;
using std::string;

class Perspective{
    public:
        Perspective( const string& persp_file);

        cv::Mat WarpImg(const cv::Mat& img, const bool& forward);

        vector<cv::Point2f> WarpPoints(const vector<cv::Point2f>& pts, const bool& forward);

        vector<cv::Point2f> GetTrapezoidPtsFromRect(const cv::Rect2d& object);
        // get 4 points of a trapezoid like  
        //  p0--p1
        // /     \
        // p3----p2


        void Draw3DBox(cv::Mat &img, const vector<cv::Point2f>& pstrapez, const cv::Scalar& color, const int& thick);


    private:

        vector<float> getLineFromAngle(cv::Point2f p1, cv::Point2f p2, cv::Point2f pc);
        
        vector<float> getLineFrom2P(cv::Point2f p1, cv::Point2f p2);
        
        cv::Point2f getInterSec(vector<float>L1,vector<float>L2);
        
        vector<cv::Point2f> getTrapezoidPts(vector<cv::Point2f> ps);
        
        void drawEllipsFromPoints(cv::Point pt0, cv::Point pt1, cv::Mat img, cv::Scalar color, int thick );


    private:
        cv::Mat matrix_;
        cv::Mat matrix_inverse_;
        int persp_width_;
        int persp_height_;
};


