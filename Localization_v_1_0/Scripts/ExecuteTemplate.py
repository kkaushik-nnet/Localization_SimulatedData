import os
from Localization_v_1_0.Scripts.DetectionPlot import detection_plot
from Localization_v_1_0.Scripts.createImageList import create_image_list
from Localization_v_1_0.Scripts.evaluateMetricPerformance_SingleMarker import evaluate_individual_performances
from Localization_v_1_0.Scripts.evaluateMetricPerformance import evaluate_distinct_data_performance
from Localization_v_1_0.Scripts.executeTestSet import execute
from Localization_v_1_0.Scripts.extractMarkerViews import extractMarkerViews
from Localization_v_1_0.Scripts.run_train_test_detections import call_create_cvs
from Localization_v_1_0.Scripts.trainAndExecuteNetwork import trainAndExecute
from Localization_v_1_0.Scripts.scaleSFAFilesize import scaleFile


def execute_template_method(train_set_items, test_set_items):
    aruco_array = train_set_items[13]
    extraction_folders = train_set_items[12]
    num_of_aruco = len(aruco_array)
    create_image_list(train_set_items[2],
                      train_set_items[4],
                      train_set_items[8])

    call_create_cvs(train_set_items[3],
                    train_set_items[1],
                    train_set_items[4],
                    train_set_items[5],
                    aruco_array)

    for var in range(num_of_aruco):
        detection_plot(train_set_items[0],
                       train_set_items[7],
                       train_set_items[5],
                       aruco_array[var],
                       'Train')

        extractMarkerViews(aruco_array[var],
                           train_set_items[5],
                           train_set_items[2],
                           train_set_items[8],
                           train_set_items[9],
                           extraction_folders[var])

        path, dirs, files = next(os.walk(extraction_folders[var]))
        extract_train_img_count = len(files)
        trainAndExecute(extraction_folders[var],
                        train_set_items[0],
                        extract_train_img_count,
                        train_set_items[8],
                        train_set_items[10],
                        aruco_array[var])

    if train_set_items[11]:
        create_image_list(test_set_items[2],
                          test_set_items[4],
                          test_set_items[8])

        call_create_cvs(test_set_items[3],
                        test_set_items[1],
                        test_set_items[4],
                        test_set_items[5],
                        aruco_array)

        for var in range(num_of_aruco):
            detection_plot(test_set_items[0],
                           test_set_items[7],
                           test_set_items[5],
                           aruco_array[var],
                           'Test')

            extractMarkerViews(aruco_array[var],
                               test_set_items[5],
                               test_set_items[2],
                               test_set_items[8],
                               test_set_items[9],
                               extraction_folders[num_of_aruco + var])

            path, dirs, files = next(os.walk(extraction_folders[num_of_aruco + var]))
            extract_test_img_count = len(files)
            execute(extraction_folders[num_of_aruco + var],
                    test_set_items[0],
                    extract_test_img_count,
                    test_set_items[8],
                    train_set_items[10],
                    aruco_array[var])

    if not train_set_items[11]:
        test_path = ''
    else:
        test_path = test_set_items[7]

    for j in range(len(train_set_items[13])):
        scaleFile(train_set_items[0], train_set_items[7], test_path,
                  aruco_array[j], train_set_items[11])
        evaluate_individual_performances(train_set_items[0], train_set_items[7], test_path,
                                         aruco_array[j], train_set_items[11])
    if len(aruco_array) > 1:
        evaluate_distinct_data_performance(train_set_items[0], train_set_items[7], test_path,
                                           train_set_items[13], train_set_items[11])
