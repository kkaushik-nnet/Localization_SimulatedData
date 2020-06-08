import os

folder_path = '/hri/localdisk/ThesisProject/MultipleDatasets/SG_datasets/Results/Results_'

for train in range(1,12):
    for test in range(1,12):
        if train != test:
            print(str(train)+'_'+str(test))
            current_working_folder = folder_path + str (train) + '_' + str (test)
            if not os.path.exists (current_working_folder):
                os.makedirs (current_working_folder)
