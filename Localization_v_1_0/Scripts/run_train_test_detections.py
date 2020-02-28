import subprocess
import os


# import ExecuteAll


def call_create_cvs(detection_path, detection_build_utils_path, xml_path, csv_path,aruco_marker_id):

    id_string = ','.join(aruco_marker_id)

    argument = (
            'sh ' + detection_path + ' ' + detection_build_utils_path + ' ' + xml_path + ' '
            + csv_path + ' ' + id_string)

    os.system(argument)


