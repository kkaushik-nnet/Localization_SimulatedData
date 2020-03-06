App_list = [
    ['Apple', 'red', 'circle'],
    ['Banana', 'yellow', 'abnormal'],
    ['Pear', 'green', 'abnormal']
]
m_1 = 160
m_2 = 150
m_3 = 140
variables = {}
for name, colour, shape in App_list:
    variables[name + "_"] = name
    variables[name + "_c"] = colour
    variables[name + "_s"] = shape

variables['Apple_n'] = 10
print(variables['Apple_n'])


