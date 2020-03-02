from twisted.python.compat import raw_input

array_global = []


def enter_array():
    num_array = []
    num = raw_input("Enter the number of Aruco_Markers present:")
    int_num = int(num)
    print('Enter numbers one by one: ')

    for i in range(int(int_num)):
        var = raw_input("num :")
        if var == '':
            print('Please enter a valid numbers again..!')
            del num_array
            break
        else:
            n = var
            num_array.append(str(n))

        if i == max(range(int_num)):
            global array_global
            array_global = num_array
            return num_array
    re_enter_array()
    return array_global


def re_enter_array():
    enter_array()


if __name__ == "__main__":
    array_ret = enter_array()
    print(array_ret)
