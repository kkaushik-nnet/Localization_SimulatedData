import os
from Localization_v_1_0.Individual_Scripts.script_main import main_program
import sys

if __name__ == "__main__":
    '''
    for train in range(1,12):
        for test in range(1,12):
            if train != test:
                main_program(train,test)
                print('#################################################################',train,'_',test)
    '''
    '''
    train = 11
    test = 10
    main_program (train, test)
    print ('#################################################################', train, '_', test)
    '''

    tr = [11,10,10,8,7,8,7,11,9,7,11]
    te = [10,11,7,7,11,9,10,7,8,8,9]
    '''
    tr = [11,10,7,8,11,9,7,11]
    te = [10,11,11,9,7,8,8,9]
    '''
    for i in range(0,len(tr)):
        train = tr[i]
        test = te[i]
        main_program ( train , test )
        print ( '#################################################################' , train , '_' , test )

