/**

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
or implied, of Rafael Muñoz Salinas.*/

#include "arucofidmarkers.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

namespace aruco
{


/**
*/
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated"
Mat FiducidalMarkers::createMarkerImage(int id, int size) throw(cv::Exception)
{
    Mat marker(size, size, CV_8UC1);
    marker.setTo(Scalar(0));
    if((0 <= id && id < 1024) || (id >= 2000 && id <= 2006)) {
        //for each line, create
        int swidth = size / 9;
        int ids[4] = {0x10, 0x17, 0x09, 0x0e};
        for(int y = 0; y < 5; y++) {
            int index = (id >> 2 * (4 - y)) & 0x0003;
            int val = ids[index];
            for(int x = 0; x < 5; x++) {
                Mat roi = marker(Rect((x + 1) * swidth, (y + 1) * swidth, swidth, swidth));
                if((val >> (4 - x)) & 0x0001) roi.setTo(Scalar(255));
                else roi.setTo(Scalar(0));
            }
        }
    } else  throw cv::Exception(9004, "id invalid", "createMarker", __FILE__, __LINE__);

    return marker;
    #pragma GCC diagnostic pop
}


/**
*
*/
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated"
cv::Mat FiducidalMarkers::getMarkerMat(int id) throw(cv::Exception)
{
    Mat marker(5, 5, CV_8UC1);
    marker.setTo(Scalar(0));
    if(0 <= id && id < 1024) {
        //for each line, create
        int ids[4] = {0x10, 0x17, 0x09, 0x0e};
        for(int y = 0; y < 5; y++) {
            int index = (id >> 2 * (4 - y)) & 0x0003;
            int val = ids[index];
            for(int x = 0; x < 5; x++) {
                if((val >> (4 - x)) & 0x0001) marker.at<uchar>(y, x) = 1;
                else marker.at<uchar>(y, x) = 0;
            }
        }
    } else throw cv::Exception(9189, "Invalid marker id", "aruco::fiducidal::createMarkerMat", __FILE__, __LINE__);
    return marker;
    #pragma GCC diagnostic pop
}


int FiducidalMarkers::analyzeMarkerImage(Mat &grey, int &nRotations, const vector<unsigned int >& markerIds)
{
    double swidth = grey.rows / 9.;

    vector<Mat> borderCandidates = vector<Mat>(4);
    borderCandidates[0] = grey(Rect(0, 0, grey.cols, int (swidth))); // top
    borderCandidates[1] = grey(Rect(0, int(swidth * 8), grey.cols, int(swidth))); // bottom
    borderCandidates[2] = grey(Rect(0, int(swidth), int(swidth), int (swidth * 7))); // left
    borderCandidates[3] = grey(Rect(int(swidth * 8), int(swidth), int(swidth), int (swidth * 7))); // right

    for(size_t i = 0; i < borderCandidates.size(); ++i) {
        int nonZero = countNonZero(borderCandidates[i]);
        if(nonZero > (borderCandidates[i].rows * borderCandidates[i].cols) / 2) {
            return -1;//can not be a marker because the border element is not black!
        }
    }
    borderCandidates.resize(8);

    // inner border and every 2nd row and column should be white
    borderCandidates[0] = grey(Rect(int(swidth), int(swidth), int(swidth * 7), int (swidth))); // top
    borderCandidates[1] = grey(Rect(int(swidth), int(swidth * 7), int(swidth * 7), int(swidth))); // bottom
    borderCandidates[2] = grey(Rect(int(swidth), int(swidth * 2), int(swidth), int (swidth * 5))); // left
    borderCandidates[3] = grey(Rect(int(swidth * 7), int(swidth * 2), int(swidth), int (swidth * 5))); // right
    borderCandidates[4] = grey(Rect(int(swidth * 2), int(swidth * 3), int(swidth * 5), int (swidth))); // 2. row
    borderCandidates[5] = grey(Rect(int(swidth * 2), int(swidth * 5), int(swidth * 5), int (swidth))); // 4. row
    borderCandidates[6] = grey(Rect(int(swidth * 3), int(swidth * 3), int(swidth), int (swidth * 5))); // 2. col
    borderCandidates[7] = grey(Rect(int(swidth * 5), int(swidth * 3), int(swidth), int (swidth * 5))); // 4. col

    for(size_t i = 0; i < borderCandidates.size(); ++i) {
        int nonZero = countNonZero(borderCandidates[i]);
        if(nonZero < (borderCandidates[i].rows * borderCandidates[i].cols) / 2) {
            return -1;//can not be a marker because the inner border element is not white!
        }
    }
    Mat bits = Mat::ones(3, 3, CV_8UC1);

    // determine for each inner square if it is  black or white
    for(int i = 0, y = 0; y < bits.rows; y++, i += 2) {
        for(int j = 0, x = 0; x < bits.cols; x++, j += 2) {
            int Xstart = int((j + 2) * swidth);
            int Ystart = int((i + 2) * swidth);

            Mat square = grey(Rect(Xstart, Ystart, swidth, swidth));
            int nZ = countNonZero(square);
            if(nZ < swidth * swidth * 0.85) {
                bits.at<uchar>(y, x) = 0;
            }
        }
    }
   // cout << bits << endl;

    // identify marker id under rotation
    // 0 degrees
    unsigned int markerId = 0;
    for(int y = 0; y < bits.rows; ++y) {
        for(int x = 0; x < bits.cols; ++x) {
            markerId <<= 1;
            markerId |= (int)bits.at<uchar>(y, x);
        }
    }
    for (size_t i = 0; i < markerIds.size(); ++i){
        if (markerId == markerIds[i]){
            nRotations = 0;
            return markerId;
        }
    }
    
    // 90 degrees
    markerId = 0;
    for(int x = 0; x < bits.cols; ++x) {
        for(int y = bits.rows - 1; y >= 0; --y) {
            markerId <<= 1;
            markerId |= (int)bits.at<uchar>(y, x);
        }
    }
    for (size_t i = 0; i < markerIds.size(); ++i){
        if (markerId == markerIds[i]){
            nRotations = 1;
            return markerId;
        }
    }

    // 180 degrees
    markerId = 0;
    for(int y = bits.rows - 1; y >= 0; --y) {
        for(int x = bits.cols - 1; x >= 0; --x) {
            markerId <<= 1;
            markerId |= (int)bits.at<uchar>(y, x);
        }
    }
    for (size_t i = 0; i < markerIds.size(); ++i){
        if (markerId == markerIds[i]){
            nRotations = 2;
            return markerId;
        }
    }

    // 270 degrees
    markerId = 0;
    for(int x = bits.cols - 1; x >= 0; --x) {
        for(int y = 0; y < bits.rows; ++y) {
            markerId <<= 1;
            markerId |= (int)bits.at<uchar>(y, x);
        }
    }
    for (size_t i = 0; i < markerIds.size(); ++i){
        if (markerId == markerIds[i]){
            nRotations = 3;
            return markerId;
        }
    }

    return -1;
}


int FiducidalMarkers::detect(const Mat &in, int &nRotations, const vector<unsigned int >& markerIds)
{
    assert(in.rows == in.cols);
    Mat grey;
    if(in.type() == CV_8UC1)
        grey = in;
    else
        cv::cvtColor(in, grey, CV_BGR2GRAY);
    //threshold image
    threshold(grey, grey, cv::mean(grey)[0] * 1.08, 255, THRESH_BINARY); // more white than black
    //threshold(grey, grey, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

    //now, analyze the interior in order to get the id
    return analyzeMarkerImage(grey, nRotations, markerIds);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated"
vector<int> FiducidalMarkers::getListOfValidMarkersIds_random(unsigned int nMarkers, vector<int> *excluded) throw(cv::Exception)
{
    if(excluded != NULL)
        if(nMarkers + excluded->size() > 1024) throw cv::Exception(8888, "FiducidalMarkers::getListOfValidMarkersIds_random", "Number of possible markers is exceeded", __FILE__, __LINE__);

    vector<int> listOfMarkers(1024);
    //set a list with all ids
    for(int i = 0; i < 1024; i++) listOfMarkers[i] = i;

    if(excluded != NULL) //set excluded to -1
        for(size_t i = 0; i < excluded->size(); i++)
            listOfMarkers[excluded->at(i)] = -1;
    //random shuffle
    random_shuffle(listOfMarkers.begin(), listOfMarkers.end());
    //now, take the first  nMarkers elements with value !=-1
    int i = 0;
    vector<int> retList;
    while(retList.size() < nMarkers) {
        if(listOfMarkers[i] != -1)
            retList.push_back(listOfMarkers[i]);
        i++;
    }
    return retList;
    #pragma GCC diagnostic pop
}

}

