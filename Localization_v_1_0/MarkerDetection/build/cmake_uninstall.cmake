# -----------------------------------------------
# File that provides "make uninstall" target
#  We use the file 'install_manifest.txt'
# -----------------------------------------------
IF(NOT EXISTS "/hri/localdisk/ThesisProject/Kaushik/Version_1_8/MarkerDetection/build/install_manifest.txt")
  MESSAGE(FATAL_ERROR "Cannot find install manifest: \"/hri/localdisk/ThesisProject/Kaushik/Version_1_8/MarkerDetection/build/install_manifest.txt\"")
ENDIF(NOT EXISTS "/hri/localdisk/ThesisProject/Kaushik/Version_1_8/MarkerDetection/build/install_manifest.txt")

FILE(READ "/hri/localdisk/ThesisProject/Kaushik/Version_1_8/MarkerDetection/build/install_manifest.txt" files)
STRING(REGEX REPLACE "\n" ";" files "${files}")
FOREACH(file ${files})
  MESSAGE(STATUS "Uninstalling \"$ENV{DESTDIR}${file}\"")
#  IF(EXISTS "$ENV{DESTDIR}${file}")
#    EXEC_PROGRAM(
#      "/hri/sit/builds/2019-05-06_09-23-31/External/CMake/3.2.2/bionic64/bin/cmake" ARGS "-E remove \"$ENV{DESTDIR}${file}\""
#      OUTPUT_VARIABLE rm_out
#      RETURN_VALUE rm_retval
#      )
	EXECUTE_PROCESS(COMMAND rm $ENV{DESTDIR}${file})
#    IF(NOT "${rm_retval}" STREQUAL 0)
#      MESSAGE(FATAL_ERROR "Problem when removing \"$ENV{DESTDIR}${file}\"")
#    ENDIF(NOT "${rm_retval}" STREQUAL 0)
#  ELSE(EXISTS "$ENV{DESTDIR}${file}")
#    MESSAGE(STATUS "File \"$ENV{DESTDIR}${file}\" does not exist.")
#  ENDIF(EXISTS "$ENV{DESTDIR}${file}")
ENDFOREACH(file)


