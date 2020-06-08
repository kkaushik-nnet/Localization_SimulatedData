x0=0
x1=2
x0_array = []
for i in range ( 0 , 20) :
    if (x1 < 20) & (x1 > 0):
        if(x1-x0) == 2:
            x1 = x1+2
            x0 = x0+2
        elif(x1-x0) == -2:
            x1 = x1-2
            x0 = x0-2
        print('cond1 --> x0: ' + str (x0) + ' - x1: ' + str (x1))
        x0_array.append(x0)
    if x1 == 20:
        x1 = x1-2
        x0 = x0+2
        print ('cond2 --> x0: ' + str (x0) + ' - x1: ' + str (x1))
        x0_array.append ( x0 )
    if x1 == 2:
        x1 = x1+2
        x0 = x0-2
        print ('cond3 --> x0: ' + str (x0) + ' - x1: ' + str (x1))
        x0_array.append ( x0 )
print(x0_array)
print(x0_array[-1])






