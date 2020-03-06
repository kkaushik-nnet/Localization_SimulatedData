import os


def call_create_cvs(detection_path, detection_build_utils_path, xml_path, csv_path,aruco_marker_id):

    id_string = ','.join(aruco_marker_id)

    argument = (
            'sh ' + os.getcwd() + '/S_run_train_detection.sh' + ' ' + detection_build_utils_path + ' ' + xml_path + ' '
            + csv_path + ' ' + id_string)

    os.system(argument)


