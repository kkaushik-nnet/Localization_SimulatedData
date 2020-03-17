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
#ifndef _Aruco_Marker_H
#define _Aruco_Marker_H
#include <vector>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "exports.h"
#include "cameraparameters.h"

using namespace std;
using namespace cv;

namespace aruco
{
/**\brief This class represents a marker. It is a vector of the fours corners ot the marker
 *
 */

class  ARUCO_EXPORTS Marker: public vector<Point2f>
{
public:
    //id of  the marker
    int id;
    // nr of clockwise 90 deg rotations
    int n_rotations;
    //size of the markers sides in meters
    float markerSizeMeters;
    // reprojection error
    double error;
    //matrices of rotation and translation with respect to the camera
    Mat Rvec, Tvec;
    vector<Point3f> markerPoints;
    vector<Point2f> markerPointsImage;

    /**
     */
    Marker();
    /**
     */
    Marker(const Marker &M);

    Marker &operator=(const Marker &M);
    
    /**Indicates if this object is valid
     */
    bool isValid()const {
        return id != -1 && size() == 4;
    }

    /**Draws this marker in the input image
     */
    void draw(Mat &in, Scalar color, int lineWidth = 1, bool writeId = true) const;

    /**Calculates the extrinsics (Rvec and Tvec) of the marker with respect to the camera
     * @param markerSize size of the marker side expressed in meters
     * @param camParams parmeters of the camera
     * @param omni flag indicating whether to use omnidirectional or perspective projection model for pose estimation.
     * @param setYPerperdicular If set the Y axis will be perpendicular to the surface. Otherwise, it will be the Z axis
     */
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated"
    void calculateExtrinsics(Mat &image, float markerSize, int canonicalMarkerSize, const CameraParameters& camParams, bool omni = true, bool setYPerperdicular = true) throw(Exception);
    #pragma GCC diagnostic pop
    /**Returns the centroid of the marker
    */
    Point2f getCenter()const;
    /**Returns the perimeter of the marker
     */
    float getPerimeter()const;
    /**Returns the area
     */
    float getArea()const;
    /**
     */
    friend bool operator<(const Marker &M1, const Marker &M2) {
        return M1.id < M2.id;
    }
    /**
     */
    friend ostream &operator<<(ostream &str, const Marker &M) {
        str << M.id << "=";
        for(int i = 0; i < 4; i++)
            str << "(" << M[i].x << "," << M[i].y << ") ";
        str << "Txyz=";
        for(int i = 0; i < 3; i++)
            str << M.Tvec.ptr<float>(0)[i] << " ";
        str << "Rxyz=";
        for(int i = 0; i < 3; i++)
            str << M.Rvec.ptr<float>(0)[i] << " ";

        return str;
    }


private:
    // Blobdetector for circular blob detection
    static Ptr<SimpleBlobDetector> blobDetector;
    static Ptr<SimpleBlobDetector> initializeBlobDetector() {
        SimpleBlobDetector::Params params = SimpleBlobDetector::Params();
        //params.minThreshold = 20;
        //params.maxThreshold = 130;
        params.filterByCircularity = true;
        params.minArea = 5;
        params.maxArea = 10000;
        params.minDistBetweenBlobs = 3;
        //params.minThreshold = 60;
        //params.maxThreshold = 120;
        //params.thresholdStep = 5;

        return SimpleBlobDetector::create(params);
    }

    void rotateXAxis(Mat &rotation);

    double computeMeanReproErr(InputArray imagePoints, InputArray proImagePoints);

    void computeJacobian(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints,
                         const Mat &K, const Mat &D, double xi, InputArray parameters,
                         Mat &JTJ_inv, Mat &JTE, double epsilon);

    void encodeParameters(InputArray R, InputArray t, OutputArray parameters);

    void decodeParameters(InputArray parameters, OutputArray R, OutputArray t);

    double initializePose(InputArray patternPoints, InputArray imagePoints, InputArray K, InputArray D,
                        double xi, OutputArray R, OutputArray t);

    double computePose(InputArray patternPoints, InputArray imagePoints, InputArray K,
                       double xi, InputArray D, OutputArray R,
                       OutputArray t, TermCriteria criteria);

};

}
#endif
