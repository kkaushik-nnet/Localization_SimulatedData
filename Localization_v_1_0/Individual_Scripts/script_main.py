import sys
from Localization_v_1_0.Individual_Scripts.Train_LabledImages import trainAndExecute
from Localization_v_1_0.Individual_Scripts.executeTestSet import execute
from Localization_v_1_0.Individual_Scripts.evaluate_multiple_object_performance import evaluate_individual_performances
from Localization_v_1_0.Individual_Scripts.evaluateMetricPerformance import evaluate_distinct_data_performance


def main_program(train,test):
    print('Hello')
    train_folder = train
    test_folder = test
    train_folder_e = train_folder*100+train_folder
    test_folder_e = test_folder*100+test_folder
    image_folder_path = '/hri/localdisk/ThesisProject/MultipleDatasets/SG_datasets/'
    jitter_image_folder_path = '/hri/localdisk/ThesisProject/MultipleDatasets/SG_datasets/Jitter/'
    hut_images = '/Extract_Hut/'
    garden_images = '/Extract_Garden_Entry/'
    building_images = '/Extract_Building_Corner/'
    output_folder = '/hri/localdisk/ThesisProject/MultipleDatasets/SG_datasets/Results/Results_'+str(train_folder)+'_'\
                    + str(test_folder)+'/'
    tags = ['Building_Corner','Hut','Garden_Entry']
    label_id = 1
    image_suffix = '.jpg'
    mode = 'train'
    # Hut
    trainAndExecute(jitter_image_folder_path + str(train_folder_e) + hut_images, output_folder, 1165, 1, image_suffix, mode)
    execute(image_folder_path + str(test_folder_e) + hut_images, output_folder, 1165, 1, image_suffix, mode)
    evaluate_individual_performances(output_folder,
                                     image_folder_path + '0' + str(train_folder) + '/coordinates_wp0.txt',
                                     image_folder_path + '0' + str(test_folder) + '/coordinates_wp0.txt',
                                     image_folder_path + '0' + str(train_folder) + '/Detection_Results.csv',
                                     image_folder_path + '0' + str(test_folder) + '/Detection_Results.csv', 1)

    trainAndExecute(jitter_image_folder_path + str(train_folder_e) + garden_images, output_folder, 0, 2, image_suffix, mode)
    execute(image_folder_path + str(test_folder_e) + garden_images, output_folder, 0, 2, image_suffix, mode)
    evaluate_individual_performances(output_folder,
                                     image_folder_path + '0' + str(train_folder) + '/coordinates_wp0.txt',
                                     image_folder_path + '0' + str(test_folder) + '/coordinates_wp0.txt',
                                     image_folder_path + '0' + str(train_folder) + '/Detection_Results.csv',
                                     image_folder_path + '0' + str(test_folder) + '/Detection_Results.csv', 2)

    trainAndExecute(jitter_image_folder_path + str(train_folder_e) + building_images, output_folder, 0, 0, image_suffix, mode)
    execute(image_folder_path + str(test_folder_e) + building_images, output_folder, 0, 0, image_suffix, mode)
    evaluate_individual_performances(output_folder,
                                     image_folder_path + '0' + str(train_folder) + '/coordinates_wp0.txt',
                                     image_folder_path + '0' + str(test_folder) + '/coordinates_wp0.txt',
                                     image_folder_path + '0' + str(train_folder) + '/Detection_Results.csv',
                                     image_folder_path + '0' + str(test_folder) + '/Detection_Results.csv', 0)

    evaluate_distinct_data_performance(output_folder,
                                       image_folder_path + '0' + str ( train_folder ) + '/coordinates_wp0.txt',
                                       image_folder_path + '0' + str ( test_folder ) + '/coordinates_wp0.txt',
                                       image_folder_path + '0' + str ( train_folder ) + '/Detection_Results.csv',
                                       image_folder_path + '0' + str ( test_folder ) + '/Detection_Results.csv',
                                       train_folder,test_folder)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python executeTestSet")
        print("     image folder        # path to the folder containing images")
        print("     total Number of Images in the folder containing images     ")
        print("     image suffix        # suffix of the image file")
        print("     Model Name          # Trained model name")
        print("\nExample: python executeTestSet.py images/ 1600 .png train\n")
        sys.exit()
    main_program(int(sys.argv[1]), int(sys.argv[2]))