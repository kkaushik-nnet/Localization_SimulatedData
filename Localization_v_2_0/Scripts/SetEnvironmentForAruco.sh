#!/bin/bash
. /hri/sit/latest/External/anaconda2/5.2/BashSrc
conda activate mypython27
cd /hri/localdisk/ThesisProject/Kaushik/Version_1_8/MarkerDetection
rm -r build
mkdir build
cd build
cmake ..
make