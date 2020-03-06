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
#ifndef _Aruco_CameraParameters_H
#define  _Aruco_CameraParameters_H

#include <opencv2/opencv.hpp>
#include "exports.h"

using namespace std;
using namespace cv;

namespace aruco
{
/**\brief Parameters of the camera
 */

class ARUCO_EXPORTS  CameraParameters
{
public:

    // 3x3 matrix (fx 0 cx, 0 fy cy, 0 0 1)
    Mat  CameraMatrix;
    //4x1 matrix (k1,k2,p1,p2)
    Mat  Distorsion;
    //size of the image
    Size CamSize;
    double xi;

    /**Empty constructor
     */
    CameraParameters() ;
    /**Creates the object from the info passed
     * @param cameraMatrix 3x3 matrix (fx 0 cx, 0 fy cy, 0 0 1)
     * @param distorsionCoeff 4x1 matrix (k1,k2,p1,p2)
     * @param size image size
     * @param xi xi parameter for omnidirectional cameras
     */
     #pragma GCC diagnostic push
     #pragma GCC diagnostic ignored "-Wdeprecated"
    CameraParameters(Mat cameraMatrix, Mat distorsionCoeff, Size size, double xi = 0) throw(Exception);
    #pragma GCC diagnostic pop
    /**Sets the parameters
     * @param cameraMatrix 3x3 matrix (fx 0 cx, 0 fy cy, 0 0 1)
     * @param distorsionCoeff 4x1 matrix (k1,k2,p1,p2)
     * @param size image size
     * @param xi xi parameter for omnidirectional cameras
     */
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated"
    void setParams(Mat cameraMatrix, Mat distorsionCoeff, Size size, double xi = 0) throw(Exception);
    #pragma GCC diagnostic pop
    /**Copy constructor
     */
    CameraParameters(const CameraParameters &CI) ;
    /**Indicates whether this object is valid
     */
    bool isValid()const {
        return CameraMatrix.rows != 0 && CameraMatrix.cols != 0  && Distorsion.rows != 0 && Distorsion.cols != 0 && CamSize.width != -1 && CamSize.height != -1;
    }
    /**Assign operator
    */
    CameraParameters &operator=(const CameraParameters &CI);
    /**Reads the camera parameters from a file generated using opencv calibration.
     */
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated"
    void readFromFile(string path)throw(Exception);
    #pragma GCC diagnostic pop
    /**Adjust the parameters to the size of the image indicated
     */
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated"
    void resize(Size size)throw(Exception);
    #pragma GCC diagnostic pop
};

}
#endif


