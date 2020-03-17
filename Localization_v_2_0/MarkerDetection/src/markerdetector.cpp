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
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <iostream>
#include <list>
#include <valarray>
#include "markerdetector.h"
#include "arucofidmarkers.h"

using namespace std;
using namespace cv;

namespace aruco
{


MarkerDetector::MarkerDetector()
{
    _doErosion = false;
    _thresMethod = ADPT_THRES;
    _thresParam1 = 7;
    _thresParam2 = 7;
    _cornerMethod = LINES;
    _markerWarpSize = 63;
    _speed = 0;
    markerIdDetector_ptrfunc = aruco::FiducidalMarkers::detect;
    pyrdown_level = 0;
    _minSize = 0.01;
    _maxSize = 0.9;
    minSize = 0;
    maxSize = 0;
}


void MarkerDetector::setDesiredSpeed(int val)
{
    if(val < 0) val = 0;
    else if(val > 3) val = 2;

    _speed = val;
    switch(_speed) {

    case 0:
        _markerWarpSize = 56;
        _cornerMethod = SUBPIX;
        _doErosion = true;
        break;

    case 1:
    case 2:
        _markerWarpSize = 28;
        _cornerMethod = NONE;
        break;
    default:
        // TODO: throw error
        break;

    };
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated"
void MarkerDetector::detect(const Mat &input, vector<Marker> &detectedMarkers, const CameraParameters& camParams, float markerSizeMeters, bool omni, bool setYPerperdicular, Rect roi) throw(Exception)
{
    // calculate the min and max contour sizes
    minSize = _minSize * max(input.cols, input.rows) * 4;
    maxSize = _maxSize * max(input.cols, input.rows) * 4;

    // convert to greyscale
    if(input.type() == CV_8UC3)
        cvtColor(input, grey, CV_BGR2GRAY);
    else
        grey = input;

    //clear input data
    detectedMarkers.clear();

    if(roi.width == 0 || roi.height == 0) {
        roi.x = 0;
        roi.y = 0;
        roi.width = grey.cols;
        roi.height = grey.rows;
    }

    Mat imgToBeThresHolded = Mat(grey, roi);
    double ThresParam1 = _thresParam1, ThresParam2 = _thresParam2;
    //Must the image be downsampled before continue processing?
    if(pyrdown_level != 0) {
        reduced = grey;
        for(int i = 0; i < pyrdown_level; i++) {
            Mat tmp;
            pyrDown(reduced, tmp);
            reduced = tmp;
        }
        int red_den = pow(2.0f, pyrdown_level);
        imgToBeThresHolded = reduced;
        ThresParam1 /= float(red_den);
        ThresParam2 /= float(red_den);
    }
    // smooth image to remove noise
    //GaussianBlur(imgToBeThresHolded, imgToBeThresHolded, Size(3, 3), 0);

    // Do threshold the image and detect contours
    thresHold(_thresMethod, imgToBeThresHolded, thres, ThresParam1, ThresParam2);

    //an erosion might be required to detect chessboard like boards
    if(_doErosion) {
        erode(thres, thres, Mat());
    }

    //find all rectangles in the thresholdes image
    vector<MarkerCandidate > MarkerCanditates;

    detectRectangles(thres, MarkerCanditates, roi.x, roi.y);

    //if the image has been downsampled, then calcualte the location of the corners in the original image
    if(pyrdown_level != 0) {
        float red_den = pow(2.0f, pyrdown_level);
        float offInc = ((pyrdown_level / 2.) - 0.5);
        for(unsigned int i = 0; i < MarkerCanditates.size(); i++) {
            for(int c = 0; c < 4; c++) {
                MarkerCanditates[i][c].x = MarkerCanditates[i][c].x * red_den + offInc;
                MarkerCanditates[i][c].y = MarkerCanditates[i][c].y * red_den + offInc;
            }
            //do the same with the the contour points
            for(unsigned int c = 0; c < MarkerCanditates[i].contour.size(); c++) {
                MarkerCanditates[i].contour[c].x = MarkerCanditates[i].contour[c].x * red_den + offInc;
                MarkerCanditates[i].contour[c].y = MarkerCanditates[i].contour[c].y * red_den + offInc;
            }
        }
    }
    // identify the markers
    for(unsigned int i = 0; i < MarkerCanditates.size(); i++) {
        // find homography
        Mat canonicalMarker;
        bool resW = warp(MarkerCanditates[i], grey, canonicalMarker, Size(_markerWarpSize, _markerWarpSize));
        if(resW) {
            int nRotations = 0;
            int id = (*markerIdDetector_ptrfunc)(canonicalMarker, nRotations, markerIds);
            if(id != -1) {
                if(_cornerMethod == LINES) refineCandidateLines(MarkerCanditates[i]); // make LINES refinement before contour points are lost
                detectedMarkers.push_back(MarkerCanditates[i]);
                detectedMarkers.back().id = id;
                detectedMarkers.back().n_rotations = nRotations;
                rotate(detectedMarkers.back().begin(), detectedMarkers.back().end() - nRotations, detectedMarkers.back().end());
            }
        }
    }

    // refine the corner location if desired
    if(detectedMarkers.size() > 0 && _cornerMethod != NONE && _cornerMethod != LINES) {
        vector<Point2f> Corners;
        for(unsigned int i = 0; i < detectedMarkers.size(); i++)
            for(int c = 0; c < 4; c++)
                Corners.push_back(detectedMarkers[i][c]);

        if(_cornerMethod == HARRIS) {
            findBestCornerInRegion_harris(grey, Corners, 7);
        } else if(_cornerMethod == SUBPIX) {
            cornerSubPix(grey, Corners, cvSize(3, 3), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 5, 0.05));
        }

        // copy back
        for(unsigned int i = 0; i < detectedMarkers.size(); i++) {
            for(int c = 0; c < 4; c++) {
                detectedMarkers[i][c] = Corners[i * 4 + c];
            }
        }
    }

    // sort by id
    sort(detectedMarkers.begin(), detectedMarkers.end());
    // there might be still the case that a marker is detected twice because of the double border indicated earlier,
    // detect and remove these cases
    vector<bool> toRemove(detectedMarkers.size(), false);
    for(int i = 0; i < int (detectedMarkers.size()) - 1; i++) {
        if(detectedMarkers[i].id == detectedMarkers[i + 1].id && !toRemove[i + 1]) {
            //deletes the one with smaller perimeter
            if(perimeter(detectedMarkers[i]) > perimeter(detectedMarkers[i + 1])) toRemove[i + 1] = true;
            else toRemove[i] = true;
        }
    }
    // remove the markers marker
    removeElements(detectedMarkers, toRemove);

    // detect the position of detected markers if desired
    if(camParams.CameraMatrix.rows != 0  && markerSizeMeters > 0) {
        // check if image size matches image size in CameraParameters
        if (input.cols == camParams.CamSize.width && input.rows == camParams.CamSize.height) {
            for(unsigned int i = 0; i < detectedMarkers.size(); i++) {
                detectedMarkers[i].calculateExtrinsics(grey, markerSizeMeters, _markerWarpSize, camParams, omni, setYPerperdicular);
            }
        }
        else {
            throw Exception(9004, "Actual image size and image size specified in camera matrix do not match!", "detect", __FILE__, __LINE__);
        }
    }
#pragma GCC diagnostic pop
}


/************************************
* 
* Crucial step. Detects the rectangular regions of the thresholded image
*
************************************/
void  MarkerDetector::detectRectangles(const Mat &thres, vector<vector<Point2f> > &MarkerCanditates, int of_x, int of_y)
{
    vector<MarkerCandidate>  candidates;
    detectRectangles(thres, candidates, of_x, of_y);
    //create the output
    MarkerCanditates.resize(candidates.size());
    for(size_t i = 0; i < MarkerCanditates.size(); i++)
        MarkerCanditates[i] = candidates[i];
}

void MarkerDetector::detectRectangles(const Mat &thresImg, vector<MarkerCandidate> & OutMarkerCanditates, int of_x, int of_y)
{
    vector<MarkerCandidate>  MarkerCanditates;
    vector<vector<Point> > contours2;

    findContours(thresImg, contours2, CV_RETR_LIST, CV_CHAIN_APPROX_NONE, Point(of_x, of_y));
    vector<Point>  approxCurve;
    
    //for each contour, analyze if it is a paralelepiped likely to be the marker
    for(unsigned int i = 0; i < contours2.size(); i++) {
        //check it is a possible element by first checking is has enough points
        if(minSize < contours2[i].size() && contours2[i].size() < maxSize) {
            //approximate to a polygon
            approxPolyDP(contours2[i], approxCurve, double(contours2[i].size()) * 0.05, true);

            //check that the polygon has 4 points
            if(approxCurve.size() == 4 && isContourConvex(Mat(approxCurve))) {
                //ensure that the distace between consecutive points is large enough
                float minDist = 1e10;
                for(int j = 0; j < 4; j++) {
                    float d = sqrt((float)(approxCurve[j].x - approxCurve[(j + 1) % 4].x) * (approxCurve[j].x - approxCurve[(j + 1) % 4].x) +
                                        (approxCurve[j].y - approxCurve[(j + 1) % 4].y) * (approxCurve[j].y - approxCurve[(j + 1) % 4].y));
                    if(d < minDist) minDist = d;
                }
                //check that distance is not very small
                if(minDist > 10) {
                    //add the points
                    MarkerCanditates.push_back(MarkerCandidate());
                    MarkerCanditates.back().idx = i;
                    MarkerCanditates.back().contour = contours2[i];
                    for(int j = 0; j < 4; j++) {
                        MarkerCanditates.back().push_back(Point2f(approxCurve[j].x, approxCurve[j].y));
                    }
                }
            }
        }
    }

    // sort the points in anti-clockwise order
    valarray<bool> swapped(false, MarkerCanditates.size());//used later

    for(size_t i = 0; i < MarkerCanditates.size(); ++i) {
        // trace a line between the first and second point.
        // if the thrid point is at the right side, then the points are anti-clockwise
        double dx1 = MarkerCanditates[i][1].x - MarkerCanditates[i][0].x;
        double dy1 =  MarkerCanditates[i][1].y - MarkerCanditates[i][0].y;
        double dx2 = MarkerCanditates[i][2].x - MarkerCanditates[i][0].x;
        double dy2 = MarkerCanditates[i][2].y - MarkerCanditates[i][0].y;
        double o = dx1 * dy2 - dy1 * dx2;

        // if the third point is in the left side, then sort in anti-clockwise order
        if(o < 0.0) {
            swap(MarkerCanditates[i][1], MarkerCanditates[i][3]);
            swapped[i] = true;
        }
    }

    // remove elements whose corners are too close to each other
    valarray<bool> toRemove(false, MarkerCanditates.size());

    for(size_t i = 0; i < MarkerCanditates.size(); i++) {
        for(size_t j = i + 1; j < MarkerCanditates.size(); j++) {
            double dist = norm(Mat(MarkerCanditates[i]), Mat(MarkerCanditates[j]));
            if(dist < 10) {
                if(perimeter(MarkerCanditates[i]) > perimeter(MarkerCanditates[i]))
                    toRemove[j] = true;
                else
                    toRemove[i] = true;
            }
        }
    }

    // remove the invalid ones
    // finally, assign to the remaining candidates the contour
    OutMarkerCanditates.reserve(MarkerCanditates.size());
    for(size_t i = 0; i < MarkerCanditates.size(); i++) {
        if(!toRemove[i]) {
            // rotate anti-clockwise if p1.x < p3.x
            if(MarkerCanditates[i][0].x - MarkerCanditates[i][2].x > 0) {
                swap(MarkerCanditates[i][3], MarkerCanditates[i][2]);
                swap(MarkerCanditates[i][2], MarkerCanditates[i][1]);
                swap(MarkerCanditates[i][1], MarkerCanditates[i][0]);
            }
            OutMarkerCanditates.push_back(MarkerCanditates[i]);
            OutMarkerCanditates.back().contour = contours2[MarkerCanditates[i].idx];
        }
    }
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated"
void MarkerDetector::thresHold(int method, const Mat &grey, Mat &out, double param1, double param2) throw(Exception)
{
    if(param1 == -1) param1 = _thresParam1;
    if(param2 == -1) param2 = _thresParam2;

    if(grey.type() != CV_8UC1) throw Exception(9001, "grey.type()!=CV_8UC1", "MarkerDetector::thresHold", __FILE__, __LINE__);
    switch(method) {
    case FIXED_THRES:
        threshold(grey, out, param1, 255, CV_THRESH_BINARY_INV);
        break;
    case ADPT_THRES://currently, this is the best method
        //ensure that _thresParam1%2==1
        if(param1 < 3) param1 = 3;
        else if(((int) param1) % 2 != 1) param1 = (int)(param1 + 1);
        adaptiveThreshold(grey, out, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, param1, param2);
        break;
    case CANNY: {
        //this should be the best method, and generally it is.
        //However, some times there are small holes in the marker contour that makes
        //the contour detector not to find it properly
        //if there is a missing pixel
        Canny(grey, out, 10, 220);
    }
        default:
        // TODO: throw error
    break;
    }
#pragma GCC diagnostic pop
}

/* TODO: chnage return type to void */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated"
bool MarkerDetector::warp(MarkerCandidate &markerCandidate, Mat &in, Mat &out, Size size) throw(Exception)
{
    if(markerCandidate.size() != 4) throw Exception(9001, "point.size()!=4", "MarkerDetector::warp", __FILE__, __LINE__);
    //obtain the perspective transform
    vector<Point2f>  pointsRes(4), pointsIn(4);
    for(int i = 0; i < 4; i++) pointsIn[i] = markerCandidate[i];
    pointsRes[0] = Point2f(0, 0);
    pointsRes[1] = Point2f(size.width - 1, 0);
    pointsRes[2] = Point2f(size.width - 1, size.height - 1);
    pointsRes[3] = Point2f(0, size.height - 1);
    Mat M = getPerspectiveTransform(&pointsIn[0], &pointsRes[0]);
    warpPerspective(in, out, M, size, INTER_NEAREST);

    return true;
    #pragma GCC diagnostic pop
}

void findCornerPointsInContour(const vector<Point2f>& points, const vector<Point> &contour, vector<int> &idxs)
{
    assert(points.size() == 4);
    int idxSegments[4] = { -1, -1, -1, -1};
    //the first point coincides with one
    Point points2i[4];
    for(int i = 0; i < 4; i++) {
        points2i[i].x = points[i].x;
        points2i[i].y = points[i].y;
    }

    for(size_t i = 0; i < contour.size(); i++) {
        if(idxSegments[0] == -1)
            if(contour[i] == points2i[0]) idxSegments[0] = i;
        if(idxSegments[1] == -1)
            if(contour[i] == points2i[1]) idxSegments[1] = i;
        if(idxSegments[2] == -1)
            if(contour[i] == points2i[2]) idxSegments[2] = i;
        if(idxSegments[3] == -1)
            if(contour[i] == points2i[3]) idxSegments[3] = i;
    }
    idxs.resize(4);
    for(int i = 0; i < 4; i++) idxs[i] = idxSegments[i];
}


bool MarkerDetector::isInto(Mat &contour, vector<Point2f> &b)
{

    for(unsigned int i = 0; i < b.size(); i++)
        if(pointPolygonTest(contour, b[i], false) > 0) return true;
    return false;
}


int MarkerDetector::perimeter(vector<Point2f> &a)
{
    int sum = 0;
    for(unsigned int i = 0; i < a.size(); i++) {
        int i2 = (i + 1) % a.size();
        sum += sqrt((a[i].x - a[i2].x) * (a[i].x - a[i2].x) + (a[i].y - a[i2].y) * (a[i].y - a[i2].y)) ;
    }
    return sum;
}


void MarkerDetector::findBestCornerInRegion_harris(const Mat   &grey, vector<Point2f> &  Corners, int blockSize)
{
    int halfSize = blockSize / 2;
    for(size_t i = 0; i < Corners.size(); i++) {
        //check that the region is into the image limits
        Point2f min(Corners[i].x - halfSize, Corners[i].y - halfSize);
        Point2f max(Corners[i].x + halfSize, Corners[i].y + halfSize);
        if(min.x >= 0  &&  min.y >= 0 && max.x < grey.cols && max.y < grey.rows) {
            // TODO: check MAt response
            Mat response;
            Mat subImage(grey, Rect(Corners[i].x - halfSize, Corners[i].y - halfSize, blockSize, blockSize));
            vector<Point2f> corners2;
            goodFeaturesToTrack(subImage, corners2, 10, 0.001, halfSize);
            float minD = 9999;
            int bIdx = -1;
            Point2f Center(halfSize, halfSize);
            for(size_t j = 0; j < corners2.size(); j++) {
                float dist = norm(corners2[j] - Center);
                if(dist < minD) {
                    minD = dist;
                    bIdx = j;
                }
                if(minD < halfSize) Corners[i] += (corners2[bIdx] - Center);
            }
        }
    }
}


void MarkerDetector::refineCandidateLines(MarkerDetector::MarkerCandidate &candidate)
{
    // search corners on the contour vector
    vector<unsigned int> cornerIndex;
    cornerIndex.resize(4);
    for(unsigned int j = 0; j < candidate.contour.size(); j++) {
        for(unsigned int k = 0; k < 4; k++) {
            if(candidate.contour[j].x == candidate[k].x && candidate.contour[j].y == candidate[k].y) {
                cornerIndex[k] = j;
            }
        }
    }

    // contour pixel in inverse order or not?
    bool inverse;
    if((cornerIndex[1] > cornerIndex[0]) && (cornerIndex[2] > cornerIndex[1] || cornerIndex[2] < cornerIndex[0]))
        inverse = false;
    else inverse = !(cornerIndex[2] > cornerIndex[1] && cornerIndex[2] < cornerIndex[0]);


    // get pixel vector for each line of the marker
    int inc = 1;
    if(inverse) inc = -1;

    vector<vector<Point> > contourLines;
    contourLines.resize(4);
    for(unsigned int l = 0; l < 4; l++) {
        for(int j = (int)cornerIndex[l]; j != (int)cornerIndex[(l + 1) % 4]; j += inc) {
            if(j == (int)candidate.contour.size() && !inverse) j = 0;
            else if(j == 0 && inverse) j = candidate.contour.size() - 1;
            contourLines[l].push_back(candidate.contour[j]);
            if(j == (int)cornerIndex[(l + 1) % 4]) break; // this has to be added because of the previous ifs
        }

    }

    // interpolate marker lines
    vector<Point3f> lines;
    lines.resize(4);
    for(unsigned int j = 0; j < lines.size(); j++) interpolate2Dline(contourLines[j], lines[j]);

    // get cross points of lines
    vector<Point2f> crossPoints;
    crossPoints.resize(4);
    for(unsigned int i = 0; i < 4; i++)
        crossPoints[i] = getCrossPoint(lines[(i - 1) % 4], lines[i]);

    // reassing points
    for(unsigned int j = 0; j < 4; j++)
        candidate[j] = crossPoints[j];
}


void MarkerDetector::interpolate2Dline(const vector< Point >& inPoints, Point3f &outLine)
{

    float minX, maxX, minY, maxY;
    minX = maxX = inPoints[0].x;
    minY = maxY = inPoints[0].y;
    for(unsigned int i = 1; i < inPoints.size(); i++)  {
        if(inPoints[i].x < minX) minX = inPoints[i].x;
        if(inPoints[i].x > maxX) maxX = inPoints[i].x;
        if(inPoints[i].y < minY) minY = inPoints[i].y;
        if(inPoints[i].y > maxY) maxY = inPoints[i].y;
    }

    // create matrices of equation system
    Mat A(inPoints.size(), 2, CV_32FC1, Scalar(0));
    Mat B(inPoints.size(), 1, CV_32FC1, Scalar(0));
    Mat X;

    if(maxX - minX > maxY - minY) {
        // Ax + C = y
        for(unsigned int i = 0; i < inPoints.size(); i++) {

            A.at<float>(i, 0) = inPoints[i].x;
            A.at<float>(i, 1) = 1.;
            B.at<float>(i, 0) = inPoints[i].y;

        }
        // solve system
        solve(A, B, X, DECOMP_SVD);
        // return Ax + By + C
        outLine = Point3f(X.at<float>(0, 0), -1., X.at<float>(1, 0));
    } else {
        // By + C = x
        for(unsigned int i = 0; i < inPoints.size(); i++) {

            A.at<float>(i, 0) = inPoints[i].y;
            A.at<float>(i, 1) = 1.;
            B.at<float>(i, 0) = inPoints[i].x;
        }
        // solve system
        solve(A, B, X, DECOMP_SVD);
        // return Ax + By + C
        outLine = Point3f(-1., X.at<float>(0, 0), X.at<float>(1, 0));
    }

}


Point2f MarkerDetector::getCrossPoint(const Point3f &line1, const Point3f &line2)
{
    // create matrices of equation system
    Mat A(2, 2, CV_32FC1, Scalar(0));
    Mat B(2, 1, CV_32FC1, Scalar(0));
    Mat X;

    A.at<float>(0, 0) = line1.x;
    A.at<float>(0, 1) = line1.y;
    B.at<float>(0, 0) = -line1.z;

    A.at<float>(1, 0) = line2.x;
    A.at<float>(1, 1) = line2.y;
    B.at<float>(1, 0) = -line2.z;

    // solve system
    solve(A, B, X, DECOMP_SVD);
    return Point2f(X.at<float>(0, 0), X.at<float>(1, 0));
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated"
void MarkerDetector::setMarkerIds(const std::vector<unsigned int >& markerIds)
{
    this->markerIds.resize(0);
    for (size_t i = 0; i < markerIds.size(); ++i){
        this->markerIds.push_back(markerIds[i]);
    }
#pragma GCC diagnostic pop
}


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated"
void MarkerDetector::setMinMaxSize(float min , float max)throw(Exception)
{
    if(min <= 0 || min > 1) throw Exception(1, " min parameter out of range", "MarkerDetector::setMinMaxSize", __FILE__, __LINE__);
    if(max <= 0 || max > 1) throw Exception(1, " max parameter out of range", "MarkerDetector::setMinMaxSize", __FILE__, __LINE__);
    if(min > max) throw Exception(1, " min>max", "MarkerDetector::setMinMaxSize", __FILE__, __LINE__);
    _minSize = min;
    _maxSize = max;
    #pragma GCC diagnostic pop
}
};

