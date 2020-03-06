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
#define _USE_MATH_DEFINES
#include <math.h>
#include <cstdio>
#include "opencv2/ccalib/omnidir.hpp"
#include "opencv2/ccalib.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "marker.h"

using namespace cv;
namespace aruco
{

Ptr<SimpleBlobDetector> Marker::blobDetector = Marker::initializeBlobDetector();


Marker::Marker()
{
    id = -1;
    n_rotations = -1;
    markerSizeMeters = -1;
    error = -1;
    Rvec.create(3, 1, CV_32FC1);
    Tvec.create(3, 1, CV_32FC1);
    for(int i = 0; i < 3; i++)
        Tvec.at<float>(i, 0) = Rvec.at<float>(i, 0) = -999999;
}

Marker::Marker(const Marker &M): vector<Point2f>(M)
{
    M.Rvec.copyTo(Rvec);
    M.Tvec.copyTo(Tvec);
    id = M.id;
    n_rotations = M.n_rotations;
    markerSizeMeters = M.markerSizeMeters;
    error = M.error;
    this->markerPoints = M.markerPoints;
    this->markerPointsImage = M.markerPointsImage;
}

Marker &Marker::operator=(const Marker &M)
{
    M.Rvec.copyTo(Rvec);
    M.Tvec.copyTo(Tvec);
    id = M.id;
    n_rotations = M.n_rotations;
    markerSizeMeters = M.markerSizeMeters;
    error = M.error;
    this->markerPoints = M.markerPoints;
    this->markerPointsImage = M.markerPointsImage;
    resize(0);
    for(unsigned int i = 0; i < M.size(); i++) {
        push_back(M[i]);
    }
    return *this;
}


void Marker::draw(Mat &in, Scalar color, int lineWidth, bool writeId) const
{
    if(size() != 4) return;
    line(in, (*this)[0], (*this)[1], color, lineWidth, CV_AA);
    line(in, (*this)[1], (*this)[2], color, lineWidth, CV_AA);
    line(in, (*this)[2], (*this)[3], color, lineWidth, CV_AA);
    line(in, (*this)[3], (*this)[0], color, lineWidth, CV_AA);
    rectangle(in, (*this)[0] - Point2f(2, 2), (*this)[0] + Point2f(2, 2), Scalar(0, 0, 255), lineWidth, CV_AA);
    rectangle(in, (*this)[1] - Point2f(2, 2), (*this)[1] + Point2f(2, 2), Scalar(0, 255, 0), lineWidth, CV_AA);
    rectangle(in, (*this)[2] - Point2f(2, 2), (*this)[2] + Point2f(2, 2), Scalar(255, 0, 0), lineWidth, CV_AA);
    rectangle(in, (*this)[3] - Point2f(2, 2), (*this)[3] + Point2f(2, 2), Scalar(0, 0, 0), lineWidth, CV_AA);
    if(writeId) {
        char cad[100];
        sprintf(cad, "id=%d", id);
        //determine the centroid
        Point cent(0, 0);
        for(int i = 0; i < 4; i++) {
            cent.x += (*this)[i].x;
            cent.y += (*this)[i].y;
        }
        cent.x /= 4.;
        cent.y /= 4.;
        putText(in, cad, cent, FONT_HERSHEY_SIMPLEX, 0.5,  Scalar(255 - color[0], 255 - color[1], 255 - color[2], 255), 2);
    }
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated"
void Marker::calculateExtrinsics(Mat &image, float markerSize, int canonicalMarkerSize, const CameraParameters &camParams, bool omni, bool setYPerperdicular) throw(Exception)
{
    if(!isValid()) throw Exception(9004, "!isValid(): invalid marker. It is not possible to calculate extrinsics", "calculateExtrinsics", __FILE__, __LINE__);
    if(camParams.CameraMatrix.rows == 0 || camParams.CameraMatrix.cols == 0) throw Exception(9004, "CameraMatrix is empty", "calculateExtrinsics", __FILE__, __LINE__);
    
    markerSizeMeters = markerSize;

    // search for blobs only in the area where the marker's rectangle was detected
    Rect roi = boundingRect(*this);

    // clip roi to image boundaries
    if(roi.x < 0) {
        roi.x = 0;
    }
    if(roi.x + roi.width >= image.cols) {
        roi.width -= (roi.x + roi.width - image.cols);
    }
    if(roi.y < 0) {
        roi.y = 0;
    }
    if(roi.y + roi.height >= image.rows) {
        roi.height -= (roi.y + roi.height - image.rows);
    }

    Mat imageMarker = image(roi);

    vector<KeyPoint> blobs;
    Marker::blobDetector->detect(imageMarker, blobs);

    // change roi coordinates to image coordinates
    for(size_t i = 0; i < blobs.size(); ++i) {
        blobs[i].pt += Point2f(roi.x, roi.y);
    }

    // create object points and object points in image by the marker's Id
    markerPoints.clear();
    markerPointsImage.clear();

    double blobSize = markerSizeMeters / 9.;
    double patternSize = (canonicalMarkerSize - 1) / 9.;
    
    // create object points from marker id for pose estimation
    for(int y = 0, i = 0; y < 3; y++, i += 2) {
        for(int x = 0, j = 0; x < 3; x++, j += 2) {
            int bitIdx = 8 - (y * 3 + x);
            if((id & (1 << bitIdx)) == 0) {
                Point3f objectPoint;
                objectPoint.x = -2. * blobSize + j * blobSize;
                objectPoint.y = 2. * blobSize - i * blobSize;
                objectPoint.z = 0;
                markerPoints.push_back(objectPoint);

                Point2f objectPointImage;
                objectPointImage.x = (j + 2.5) * patternSize;
                objectPointImage.y = (i + 2.5) * patternSize;
                markerPointsImage.push_back(objectPointImage);
            }
        }
    }
    
    // transform object image points by the marker's homography
    vector<Point2f> markerPointsImageTransformed;
    vector<Point2f>  canonicalPoints(4);
    canonicalPoints[0] = Point2f(0, 0);
    canonicalPoints[1] = Point2f(canonicalMarkerSize - 1, 0);
    canonicalPoints[2] = Point2f(canonicalMarkerSize - 1, canonicalMarkerSize - 1);
    canonicalPoints[3] = Point2f(0, canonicalMarkerSize - 1);
    Mat H = getPerspectiveTransform(&canonicalPoints[0], &(*this)[0]);
    perspectiveTransform(markerPointsImage, markerPointsImageTransformed, H);
    
    // filter out blobs too far away from closest projected point
    double max_side_len = 0;
    // determine max side length of marker contour
    for(unsigned long i = 0; i < 3; i++) {
        double side_len = norm(at(i) - at(i + 1));
        if(side_len > max_side_len) {
            max_side_len = side_len;
        }
    }
    double distThreshold = max_side_len / 9.;

    markerPointsImage.clear();
    int idxBestMatch;

    /* TODO: check for double assignment */
    for(size_t i = 0; i < markerPointsImageTransformed.size(); ++i) {
        double minDist = DBL_MAX;
        for(size_t j = 0; j < blobs.size(); ++j) {
            double distx = markerPointsImageTransformed[i].x - blobs[j].pt.x;
            double disty = markerPointsImageTransformed[i].y - blobs[j].pt.y;

            double dist = sqrt(distx * distx + disty * disty);

            if(dist < minDist) {
                minDist = dist;
                idxBestMatch = j;
            }
        }
        if(minDist < distThreshold) {
            markerPointsImage.push_back(blobs[idxBestMatch].pt);
        }
    }
    
    if(markerPoints.size() > 3 && markerPoints.size() == markerPointsImage.size()) {
        // Right now the corners of the marker are not used because corner location is imprecise
        // add corner points of marker rectangle
// 	markerPoints.resize(0);
// 	markerPointsImage.resize(0);
// 	markerPoints.push_back(Point3f(-markerSizeMeters / 2, markerSizeMeters / 2, 0));
// 	markerPoints.push_back(Point3f(markerSizeMeters / 2, markerSizeMeters / 2, 0));
// 	markerPoints.push_back(Point3f(markerSizeMeters / 2, -markerSizeMeters / 2, 0));
// 	markerPoints.push_back(Point3f(-markerSizeMeters / 2, -markerSizeMeters / 2, 0));
// 	markerPointsImage.push_back(at(0));
// 	markerPointsImage.push_back(at(1));
// 	markerPointsImage.push_back(at(2));
// 	markerPointsImage.push_back(at(3));

        Mat raux, taux;
        
        if(omni) {
            TermCriteria criteria(3, 300, 1e-9);
            error = computePose(markerPoints, markerPointsImage, camParams.CameraMatrix, camParams.xi, camParams.Distorsion, raux, taux, criteria);
            raux.convertTo(Rvec, CV_32F);
            taux.convertTo(Tvec, CV_32F);
        } else {
            if(solvePnP(markerPoints, markerPointsImage, camParams.CameraMatrix, camParams.Distorsion, raux, taux)) {
                Mat projectedImagePoints;
                cv::projectPoints(markerPoints, raux, taux, camParams.CameraMatrix, camParams.Distorsion, projectedImagePoints);
                error = norm(Mat(markerPointsImage) - projectedImagePoints);
                
                raux.convertTo(Rvec, CV_32F);
                taux.convertTo(Tvec, CV_32F);
            } else {
                id = -1;
            }
        }
        //rotate the X axis so that Y is perpendicular to the marker plane
        if(setYPerperdicular) {
            rotateXAxis(Rvec);
        }
    } else {
        id = -1;
    }
#pragma GCC diagnostic pop
}

void Marker::rotateXAxis(Mat &rotation)
{
    Mat R(3, 3, CV_32F);
    Rodrigues(rotation, R);
    //create a rotation matrix for x axis
    Mat RX = Mat::eye(3, 3, CV_32F);
    //float angleRad = M_PI / 2;
    RX.at<float>(1, 1) = 0; //cos(angleRad);
    RX.at<float>(1, 2) = -1; //-sin(angleRad);
    RX.at<float>(2, 1) = 1; //sin(angleRad);
    RX.at<float>(2, 2) = 0; //cos(angleRad);
    //now multiply
    R = R * RX;
    //finally, the the rodrigues back
    Rodrigues(R, rotation);
}

Point2f Marker::getCenter() const
{
    Point2f center(0, 0);
    for(size_t i = 0; i < size(); i++) {
        center.x += (*this)[i].x;
        center.y += (*this)[i].y;
    }
    center.x /= size();
    center.y /= size();
    return center;
}

float Marker::getArea() const
{
    assert(size() == 4);
    //use the cross products
    Point2f v01 = (*this)[1] - (*this)[0];
    Point2f v03 = (*this)[3] - (*this)[0];
    float area1 = fabs(v01.x * v03.y - v01.y * v03.x);
    Point2f v21 = (*this)[1] - (*this)[2];
    Point2f v23 = (*this)[3] - (*this)[2];
    float area2 = fabs(v21.x * v23.y - v21.y * v23.x);
    return (area2 + area1) / 2.;


}

float Marker::getPerimeter() const
{
    assert(size() == 4);
    float sum = 0;
    for(int i = 0; i < 4; i++)
        sum += norm((*this)[i] - (*this)[(i + 1) % 4]);
    return sum;
}


double Marker::computeMeanReproErr(InputArray imagePoints, InputArray projectedImagePoints)
{
    CV_Assert(!imagePoints.empty() && imagePoints.type() == CV_64FC2);
    CV_Assert(!projectedImagePoints.empty() && projectedImagePoints.type() == CV_64FC2);
    CV_Assert(imagePoints.total() == projectedImagePoints.total());

    int n = (int)imagePoints.total();
    double reprojectionError = 0;
    int totalPoints = 0;
    if(imagePoints.kind() == _InputArray::STD_VECTOR_MAT) {
        for(int i = 0; i < n; i++) {
            Mat x, proj_x;
            imagePoints.getMat(i).copyTo(x);
            projectedImagePoints.getMat(i).copyTo(proj_x);
            Mat errorI = x.reshape(2, x.rows * x.cols) - proj_x.reshape(2, proj_x.rows * proj_x.cols);
            totalPoints += (int)errorI.total();
            Vec2d *ptr_err = errorI.ptr<Vec2d>();
            for(int j = 0; j < (int)errorI.total(); j++) {
                reprojectionError += sqrt(ptr_err[j][0] * ptr_err[j][0] + ptr_err[j][1] * ptr_err[j][1]);
            }
        }
    } else {
        Mat x, proj_x;
        imagePoints.getMat().copyTo(x);
        projectedImagePoints.getMat().copyTo(proj_x);
        Mat errorI = x.reshape(2, x.rows * x.cols) - proj_x.reshape(2, proj_x.rows * proj_x.cols);
        totalPoints += (int)errorI.total();
        Vec2d *ptr_err = errorI.ptr<Vec2d>();
        for(int j = 0; j < (int)errorI.total(); j++) {
            reprojectionError += sqrt(ptr_err[j][0] * ptr_err[j][0] + ptr_err[j][1] * ptr_err[j][1]);
        }
    }
    return reprojectionError / totalPoints;
}


void Marker::computeJacobian(InputArray objectPoints, InputArray imagePoints,
                             const Mat &K, const Mat &D, double xi, InputArray parameters,
                             Mat &JTJ_inv, Mat &JTE, double epsilon)
{
    CV_Assert(!objectPoints.empty() && objectPoints.type() == CV_64FC3);
    CV_Assert(!imagePoints.empty() && imagePoints.type() == CV_64FC2);

    Mat JTJ = Mat::zeros(6, 6, CV_64F);
    JTJ_inv = Mat::zeros(6, 6, CV_64F);
    JTE = Mat::zeros(6, 1, CV_64F);

    int nPointsAll = (int)objectPoints.getMat().total();

    Mat J = Mat::zeros(2 * nPointsAll, 6, CV_64F);
    Mat exAll = Mat::zeros(2 * nPointsAll, 6, CV_64F);

    Mat objPoints, imgPoints, om, T;
    objectPoints.getMat().copyTo(objPoints);
    imagePoints.getMat().copyTo(imgPoints);
    objPoints = objPoints.reshape(3, objPoints.rows * objPoints.cols);
    imgPoints = imgPoints.reshape(2, imgPoints.rows * imgPoints.cols);

    om = parameters.getMat().colRange(0, 3);
    T = parameters.getMat().colRange(3, 6);
    Mat imgProj, jacobian;
    omnidir::projectPoints(objPoints, imgProj, om, T, K, xi, D, jacobian);
    Mat projError = imgPoints - imgProj;

    Mat JEx(jacobian.rows, 6, CV_64F);

    jacobian.colRange(0, 6).copyTo(JEx);

    JTJ(Rect(0, 0, 6, 6)) = JEx.t() * JEx;
    JTE(Rect(0, 0, 1, 6)) = JEx.t() * projError.reshape(1, 2 * (int)projError.total());

    JTJ_inv = Mat(JTJ + epsilon).inv();
}

void Marker::encodeParameters(InputArray R, InputArray t, OutputArray parameters)
{
    Mat _R = R.getMat(), _t = t.getMat();

    parameters.create(1, 6, CV_64F);
    Mat _params = parameters.getMat();
    Mat(_R).reshape(1, 1).copyTo(_params.colRange(0, 3));
    Mat(_t).reshape(1, 1).copyTo(_params.colRange(3, 6));
}

void Marker::decodeParameters(InputArray parameters, OutputArray R, OutputArray t)
{
    Mat _params = parameters.getMat();

    Vec3d _R, _t;
    _R = Vec3d(_params.colRange(0, 3));
    _t = Vec3d(_params.colRange(3, 6));

    Mat(_R).convertTo(R, CV_64FC3);
    Mat(_t).convertTo(t, CV_64FC3);
}


double Marker::initializePose(InputArray patternPoints, InputArray imagePoints, InputArray K, InputArray D,
                              double xi, OutputArray R, OutputArray t)
{
    // For details please refer to Section III from Li's IROS 2013 paper
    Mat K_ = K.getMat();
    double u0 = K_.at<double>(0, 2);
    double v0 = K_.at<double>(1, 2);
    double fx = K_.at<double>(0, 0);
    
    Vec3d v_R, v_t;

    Mat objPoints, imgPoints;
    patternPoints.getMat().copyTo(objPoints);
    imagePoints.getMat().copyTo(imgPoints);

    int n_point = imgPoints.rows * imgPoints.cols;
    if(objPoints.rows != n_point)
        objPoints = objPoints.reshape(3, n_point);
    if(imgPoints.rows != n_point)
        imgPoints = imgPoints.reshape(2, n_point);

    // objectPoints should be 3-channel data, imagePoints should be 2-channel data
    CV_Assert(objPoints.type() == CV_64FC3 && imgPoints.type() == CV_64FC2);

    vector<Mat> xy, uv;
    split(objPoints, xy);
    split(imgPoints, uv);

    Mat x = xy[0].reshape(1, n_point), y = xy[1].reshape(1, n_point),
        u = uv[0].reshape(1, n_point) - u0, v = uv[1].reshape(1, n_point) - v0;

    Mat sqrRho = u.mul(u) + v.mul(v);

    // compute extrinsic parameters
    Mat M(n_point, 6, CV_64F);
    Mat(-v.mul(x)).copyTo(M.col(0));
    Mat(-v.mul(y)).copyTo(M.col(1));
    Mat(u.mul(x)).copyTo(M.col(2));
    Mat(u.mul(y)).copyTo(M.col(3));
    Mat(-v).copyTo(M.col(4));
    Mat(u).copyTo(M.col(5));

    Mat W, U, V;
    SVD::compute(M, W, U, V, SVD::FULL_UV);
    V = V.t();

    double miniReprojectError = 1e5;
    // the signs of r1, r2, r3 are unknown, so they can be flipped.
    for(int coef = 1; coef >= -1; coef -= 2) {
        double r11 = V.at<double>(0, 5) * coef;
        double r12 = V.at<double>(1, 5) * coef;
        double r21 = V.at<double>(2, 5) * coef;
        double r22 = V.at<double>(3, 5) * coef;
        double t1 = V.at<double>(4, 5) * coef;
        double t2 = V.at<double>(5, 5) * coef;

        Mat roots;
        double r31s;
        solvePoly(Matx13d(-(r11 * r12 + r21 * r22) * (r11 * r12 + r21 * r22), r11 * r11 + r21 * r21 - r12 * r12 - r22 * r22, 1), roots);

        if(roots.at<Vec2d>(0)[0] > 0)
            r31s = sqrt(roots.at<Vec2d>(0)[0]);
        else
            r31s = sqrt(roots.at<Vec2d>(1)[0]);

        for(int coef2 = 1; coef2 >= -1; coef2 -= 2) {
            double r31 = r31s * coef2;
            double r32 = -(r11 * r12 + r21 * r22) / r31;

            Vec3d r1(r11, r21, r31);
            Vec3d r2(r12, r22, r32);
            Vec3d t(t1, t2, 0);
            double scale = 1 / norm(r1);
            r1 = r1 * scale;
            r2 = r2 * scale;
            t = t * scale;

            Vec3d r3 = r1.cross(r2);
            // compute t3 from equation in paper "A Multiple Camera System Calibration Toolbox Using A Feature ..."
            Mat p(n_point, 1, CV_64F); // p**2
            for(int i = 0; i < n_point; ++i) {
                p.at<double>(i, 0) = u.at<double>(i, 0) * u.at<double>(i, 0) + v.at<double>(i, 0) * v.at<double>(i, 0);
            }
            Mat f_u_v = fx / 2. - 1. / (2. * fx) * p; // fx or fy ??
            Mat t3(n_point * 2, 1, CV_64F), t3_1, t3_2;

            divide(v.mul(r3[0] * x + r3[1] * y) - f_u_v.mul(r2[0] * x + r2[1] * y + t[1]), -v, t3_1);
            divide(f_u_v.mul(r1[0] * x + r2[0] * y + t[0]) - u.mul(r3[0] * x + r3[1] * y), u, t3_2);
            
            t3_1.rowRange(0, n_point).copyTo(t3.rowRange(0, n_point));
            t3_2.rowRange(0, n_point).copyTo(t3.rowRange(n_point, n_point * 2));

            Matx33d R(r1[0], r2[0], r3[0],
                      r1[1], r2[1], r3[1],
                      r1[2], r2[2], r3[2]);
            Vec3d om;
            Rodrigues(R, om);

            // chose t3 which minimizes reprojection error
            double reprojectError = 1e6;
            double final_t3 = 0;

            for(int i = 0; i < t3.rows; ++i) {
                t[2] = t3.at<double>(i);
                // compute reprojection error
                Mat projedImgPoints;
                omnidir::projectPoints(objPoints, projedImgPoints, om, t, K, xi, D, noArray());
                double currentReprojectError = computeMeanReproErr(imgPoints, projedImgPoints);

                if(currentReprojectError < reprojectError) {
                    final_t3 = t3.at<double>(i);
                    reprojectError = currentReprojectError;
                }
            }
            t[2] = final_t3;

            if(reprojectError < miniReprojectError) {
                miniReprojectError = reprojectError;
                v_R = om;
                v_t = t;
            }
        }
    }

    Mat(v_R).convertTo(R, CV_64FC3);
    Mat(v_t).convertTo(t, CV_64FC3);

    return miniReprojectError;
}


double Marker::computePose(InputArray patternPoints, InputArray imagePoints, InputArray K,
                           double xi, InputArray D, OutputArray R,
                           OutputArray t, TermCriteria criteria)
{
    CV_Assert(!patternPoints.empty() && !imagePoints.empty() && patternPoints.total() == imagePoints.total());
    CV_Assert((patternPoints.type() == CV_64FC3 && imagePoints.type() == CV_64FC2) ||
              (patternPoints.type() == CV_32FC3 && imagePoints.type() == CV_32FC2));
    CV_Assert(patternPoints.getMat().channels() == 3 && imagePoints.getMat().channels() == 2);
    CV_Assert(!K.empty() && K.size() == Size(3, 3));
    CV_Assert(!D.empty() && D.total() == 4);
    int depth = patternPoints.depth();

    Mat _patternPoints, _imagePoints;

    if(depth == CV_32F) {
        patternPoints.getMat().convertTo(_patternPoints, CV_64FC3);
        imagePoints.getMat().convertTo(_imagePoints, CV_64FC2);
    }

    // initialization
    Mat _R, _t;
    initializePose(_patternPoints, _imagePoints, K, D, xi, _R, _t);

    // optimization
    Mat finalParam(1, 6, CV_64F);
    Mat currentParam(1, 6, CV_64F);
    encodeParameters(_R, _t, currentParam);
    
    const double alpha_smooth = 0.01;
    double change = 1;
    
    for(int iter = 0; ; ++iter) {
        if((criteria.type == 1 && iter >= criteria.maxCount)  ||
                (criteria.type == 2 && change <= criteria.epsilon) ||
                (criteria.type == 3 && (change <= criteria.epsilon || iter >= criteria.maxCount)))
            break;
        double alpha_smooth2 = 1 - pow(1 - alpha_smooth, (double)iter + 1.0);
        Mat JTJ_inv, JTError;
        double epsilon = 0.01 * pow(0.9, (double)iter / 10);

        computeJacobian(_patternPoints, _imagePoints, K.getMat(), D.getMat(), xi, currentParam, JTJ_inv, JTError, epsilon);

        // Gauss Newton
        Mat G = alpha_smooth2 * JTJ_inv * JTError;

        finalParam = currentParam + G.t();

        change = norm(G) / norm(currentParam);

        currentParam = finalParam.clone();

        decodeParameters(currentParam, _R, _t);
    }
    
    decodeParameters(currentParam, _R, _t);

    Mat(_R).convertTo(R, CV_64FC3);
    Mat(_t).convertTo(t, CV_64FC3);
    
    // compute and return final reprojection error
    Mat projectedImagePoints;
    omnidir::projectPoints(_patternPoints, projectedImagePoints, R, t, K, xi, D, noArray());
    return computeMeanReproErr(_imagePoints, projectedImagePoints);
}

}
