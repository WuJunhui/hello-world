#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include "perspective.h"
using std::vector;
using std::string;

Perspective::Perspective(const string& persp_file){
    vector<cv::Point2f> src_pts, dst_pts;
    double x,y;
    std::ifstream fileStream(persp_file);
    for(int i=0; i<4;i++){
        fileStream>>x;
        fileStream>>y;
        src_pts.push_back(cv::Point2f(x,y));
    }
    for(int i=0; i<4;i++){
        fileStream>>x;
        fileStream>>y;
        dst_pts.push_back(cv::Point2f(x,y));
    }

    fileStream>>persp_width_;
    fileStream>>persp_height_;

    matrix_= cv::getPerspectiveTransform(src_pts, dst_pts);
    matrix_inverse_ = cv::getPerspectiveTransform(dst_pts, src_pts);
}

cv::Mat Perspective::WarpImg(const cv::Mat& img, const bool& forward){
    cv::Mat warped_img;
    if (forward)
        cv::warpPerspective(img, warped_img, matrix_, cv::Size(persp_width_, persp_height_));

    else
        cv::warpPerspective(img, warped_img, matrix_inverse_, cv::Size(persp_width_, persp_height_));

    return warped_img;
}


vector<cv::Point2f> Perspective::WarpPoints(const vector<cv::Point2f>& pts, const bool& forward){
    vector<cv::Point2f> pts2;
    if (forward)
        perspectiveTransform(pts, pts2,  matrix_);
    else
        perspectiveTransform(pts, pts2,  matrix_inverse_);

    return pts2;
}



vector<float> Perspective::getLineFromAngle(cv::Point2f p1, cv::Point2f p2, cv::Point2f pc){//line perp to line(p1p2), and pass by pc
    vector<float> lineABC;
    if (p1.y-p2.y == 0){
        lineABC.push_back(1.0);
        lineABC.push_back(0.0);
        lineABC.push_back(pc.x);
    }
    else {
        float tan = -(p1.x-p2.x) / (p1.y-p2.y);
        lineABC.push_back(-tan);
        lineABC.push_back(1.0);
        lineABC.push_back(pc.y-pc.x*tan);
    }
    return lineABC;
}

vector<float> Perspective::getLineFrom2P(cv::Point2f p1, cv::Point2f p2){
    vector<float> lineABC;
    lineABC.push_back(p1.y-p2.y);
    lineABC.push_back(p2.x-p1.x);
    lineABC.push_back(-p1.x*p2.y + p2.x*p1.y);
    return lineABC;
}

cv::Point2f Perspective::getInterSec(vector<float>L1,vector<float>L2){
    float D = L1[0] * L2[1] - L1[1] * L2[0];
    float Dx = L1[2] * L2[1] - L1[1] * L2[2];
    float Dy = L1[0] * L2[2] - L1[2] * L2[0];
    if (D !=0)
        return cv::Point2f(Dx/D, Dy/D);
    else
        return cv::Point2f(-1000,-1000); 
}

vector<cv::Point2f> Perspective::getTrapezoidPts(vector<cv::Point2f> ps){// from 4pts 
    // p0---p1
    // |    |
    // p3---p2
    vector<cv::Point2f> pstrapez;
    cv::Point2f pu = (ps[0]+ps[1])*0.5;
    cv::Point2f pb = (ps[2]+ps[3])*0.5;
    vector<float> Lu=getLineFromAngle(pu,pb,pu);
    vector<float> Lb=getLineFromAngle(pu,pb,pb);
    vector<float> Ll = getLineFrom2P(ps[0],ps[3]);
    vector<float> Lr = getLineFrom2P(ps[1],ps[2]);

    pstrapez.push_back( getInterSec(Lu,Ll));    
    pstrapez.push_back( getInterSec(Lu,Lr));    
    pstrapez.push_back( getInterSec(Lb,Lr));    
    pstrapez.push_back( getInterSec(Lb,Ll));    
    return pstrapez;
}


vector<cv::Point2f> Perspective::GetTrapezoidPtsFromRect(const cv::Rect2d& object){
    vector<cv::Point2f> pts_on_warped_img;
    pts_on_warped_img.push_back(cv::Point(object.x,object.y));
    pts_on_warped_img.push_back(cv::Point(object.x+object.width,object.y));
    pts_on_warped_img.push_back(cv::Point(object.x+object.width,object.y+object.height));
    pts_on_warped_img.push_back(cv::Point(object.x,object.y+object.height));

    vector<cv::Point2f> pts_on_img_before_warped;
    perspectiveTransform(pts_on_warped_img, pts_on_img_before_warped, matrix_inverse_);
    vector<cv::Point2f> pstrapez = getTrapezoidPts(pts_on_img_before_warped);
    return pstrapez;
}

void Perspective::drawEllipsFromPoints(cv::Point pt0, cv::Point pt1, cv::Mat img, cv::Scalar color, int thick =1){
    double pi = 3.1415926535897;
    float angle = atan2(pt1.y - pt0.y, 
        pt1.x - pt0.x) *180 /pi; //TODO: PI

    float elli_l=sqrt( (pt0-pt1).ddot(pt0-pt1) )*0.5;
    float elli_s=elli_l*0.3;

    cv::ellipse(img, (pt0 + pt1)*0.5, 
        cv::Size(elli_l,elli_s),
        angle,
        0,
        360,
        color,
        thick);
}

void Perspective::Draw3DBox(cv::Mat &img, const vector<cv::Point2f>& pstrapez, const cv::Scalar& color, const int& thick=1){
        cv::line(img, pstrapez[1],pstrapez[2],color,thick);
        cv::line(img, pstrapez[3],pstrapez[0],color,thick);
        drawEllipsFromPoints(pstrapez[0],pstrapez[1],img,color,thick);
        drawEllipsFromPoints(pstrapez[3],pstrapez[2],img,color,thick);
}
