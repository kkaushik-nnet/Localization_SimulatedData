#!/bin/bash
Path_TO_EXECUTABLE=$1
Image_NAME_List_Path=$2
Output_Path=$3
Intrinsics_File_Path="/hri/localdisk/ThesisProject/Kaushik/Simulator_data/intrinsics_2880x2880.yml"
Marker_ID=[$4]
Marker_Size=1.
Is_Omni=1
Debug=0

cd $Path_TO_EXECUTABLE
./detect_markers $Image_NAME_List_Path $Output_Path $Intrinsics_File_Path $Marker_ID $Marker_Size $Is_Omni $Debug

