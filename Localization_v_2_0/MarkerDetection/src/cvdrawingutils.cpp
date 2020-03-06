/*****************************
Copyright 2011 Rafael Muñoz Salinas. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY Rafael Muñoz Salinas ''AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Rafael Muñoz Salinas OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of Rafael Muñoz Salinas.
********************************/
#include "cvdrawingutils.h"

using namespace std;
using namespace cv;

namespace aruco
{

void CvDrawingUtils::draw3dAxis(Mat &Image, Marker &m, const CameraParameters &CP, bool omni)
{
    float size = m.markerSizeMeters * 3;
    Mat objectPoints(4, 3, CV_32FC1);
    objectPoints.at<float>(0, 0) = 0;
    objectPoints.at<float>(0, 1) = 0;
    objectPoints.at<float>(0, 2) = 0;
    objectPoints.at<float>(1, 0) = size;
    objectPoints.at<float>(1, 1) = 0;
    objectPoints.at<float>(1, 2) = 0;
    objectPoints.at<float>(2, 0) = 0;
    objectPoints.at<float>(2, 1) = size;
    objectPoints.at<float>(2, 2) = 0;
    objectPoints.at<float>(3, 0) = 0;
    objectPoints.at<float>(3, 1) = 0;
    objectPoints.at<float>(3, 2) = size;

    vector<Point2f> imagePoints;
    if (omni){
        objectPoints = objectPoints.reshape(3, 4);
        if(m.Tvec.cols == 1 && m.Tvec.rows == 1) {
            Mat R_ = (Mat_<float>(3, 1) << m.Rvec.at<float>(0, 0), m.Rvec.at<float>(0, 1), m.Rvec.at<float>(0, 2));
            Mat t_ = (Mat_<float>(3, 1) << m.Tvec.at<float>(0, 0), m.Tvec.at<float>(0, 1), m.Tvec.at<float>(0, 2));
            omnidir::projectPoints(objectPoints, imagePoints, R_, t_, CP.CameraMatrix, CP.xi, CP.Distorsion);
        } else {
            omnidir::projectPoints(objectPoints, imagePoints, m.Rvec, m.Tvec, CP.CameraMatrix, CP.xi, CP.Distorsion);
        }
    }
    else{
        cv::projectPoints(objectPoints, m.Rvec, m.Tvec, CP.CameraMatrix, CP.Distorsion, imagePoints);
    }

    if(Image.type() == CV_8UC1) {
        Scalar color(128);
        line(Image, imagePoints[0], imagePoints[1], color, 1, CV_AA);
        line(Image, imagePoints[0], imagePoints[2], color, 1, CV_AA);
        line(Image, imagePoints[0], imagePoints[3], color, 1, CV_AA);
        putText(Image, "x", imagePoints[1], FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        putText(Image, "y", imagePoints[2], FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
        putText(Image, "z", imagePoints[3], FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    } else {
        line(Image, imagePoints[0], imagePoints[1], Scalar(0, 0, 255, 255), 1, CV_AA);
        line(Image, imagePoints[0], imagePoints[2], Scalar(0, 255, 0, 255), 1, CV_AA);
        line(Image, imagePoints[0], imagePoints[3], Scalar(255, 0, 0, 255), 1, CV_AA);
        putText(Image, "x", imagePoints[1], FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255, 255), 2);
        putText(Image, "y", imagePoints[2], FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0, 255), 2);
        putText(Image, "z", imagePoints[3], FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0, 255), 2);
    }
}


void CvDrawingUtils::draw3dCube(Mat &Image, Marker &m, const CameraParameters &CP, bool omni)
{
    Mat objectPoints(8, 3, CV_32FC1);
    float halfSize = m.markerSizeMeters / 2;
    objectPoints.at<float>(0, 0) = -halfSize;
    objectPoints.at<float>(0, 1) = 0;
    objectPoints.at<float>(0, 2) = -halfSize;
    objectPoints.at<float>(1, 0) = halfSize;
    objectPoints.at<float>(1, 1) = 0;
    objectPoints.at<float>(1, 2) = -halfSize;
    objectPoints.at<float>(2, 0) = halfSize;
    objectPoints.at<float>(2, 1) = 0;
    objectPoints.at<float>(2, 2) = halfSize;
    objectPoints.at<float>(3, 0) = -halfSize;
    objectPoints.at<float>(3, 1) = 0;
    objectPoints.at<float>(3, 2) = halfSize;

    objectPoints.at<float>(4, 0) = -halfSize;
    objectPoints.at<float>(4, 1) = m.markerSizeMeters;
    objectPoints.at<float>(4, 2) = -halfSize;
    objectPoints.at<float>(5, 0) = halfSize;
    objectPoints.at<float>(5, 1) = m.markerSizeMeters;
    objectPoints.at<float>(5, 2) = -halfSize;
    objectPoints.at<float>(6, 0) = halfSize;
    objectPoints.at<float>(6, 1) = m.markerSizeMeters;
    objectPoints.at<float>(6, 2) = halfSize;
    objectPoints.at<float>(7, 0) = -halfSize;
    objectPoints.at<float>(7, 1) = m.markerSizeMeters;
    objectPoints.at<float>(7, 2) = halfSize;

    vector<Point2f> imagePoints;
    if (omni){
        objectPoints = objectPoints.reshape(3, 8);
        if(m.Tvec.cols == 1 && m.Tvec.rows == 1) {
            Mat R_ = (Mat_<float>(3, 1) << m.Rvec.at<float>(0, 0), m.Rvec.at<float>(0, 1), m.Rvec.at<float>(0, 2));
            Mat t_ = (Mat_<float>(3, 1) << m.Tvec.at<float>(0, 0), m.Tvec.at<float>(0, 1), m.Tvec.at<float>(0, 2));
            omnidir::projectPoints(objectPoints, imagePoints, R_, t_, CP.CameraMatrix, CP.xi, CP.Distorsion);
        } else {
            omnidir::projectPoints(objectPoints, imagePoints, m.Rvec, m.Tvec, CP.CameraMatrix, CP.xi, CP.Distorsion);
        }
    }
    else{
        cv::projectPoints(objectPoints, m.Rvec, m.Tvec, CP.CameraMatrix, CP.Distorsion, imagePoints);
    }

    if(Image.type() == CV_8UC1) {
        Scalar color(128);
        for(int i = 0; i < 4; i++)
            line(Image, imagePoints[i], imagePoints[(i + 1) % 4], color, 1, CV_AA);

        for(int i = 0; i < 4; i++)
            line(Image, imagePoints[i + 4], imagePoints[4 + (i + 1) % 4], color, 1, CV_AA);

        for(int i = 0; i < 4; i++)
            line(Image, imagePoints[i], imagePoints[i + 4], color, 1, CV_AA);
    } else {
        for(int i = 0; i < 4; i++)
            line(Image, imagePoints[i], imagePoints[(i + 1) % 4], Scalar(0, 0, 255, 255), 1, CV_AA);

        for(int i = 0; i < 4; i++)
            line(Image, imagePoints[i + 4], imagePoints[4 + (i + 1) % 4], Scalar(0, 0, 255, 255), 1, CV_AA);

        for(int i = 0; i < 4; i++)
            line(Image, imagePoints[i], imagePoints[i + 4], Scalar(0, 0, 255, 255), 1, CV_AA);
    }
}

}
