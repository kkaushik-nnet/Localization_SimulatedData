import numpy as np

pickleData = np.load('data.pickle')
fh = open('log.txt','wt')


for item in pickleData:
    if u'cameraName' in item and item[u'cameraName'] == 'kodak':
        fh.write("receiveTimeStampKodak=" + str(item[u't']) + "," + "frameNameKodak=" + str(item[u'frameName']) + '\n')
        fh.write("*********\n")
    if u'cameraName' in item and item[u'cameraName'] == 'brio':
        fh.write("receiveTimeStampBrio=" + str(item[u't']) + "," + "frameNameBrio=" + str(item[u'frameName']) + '\n')
        fh.write("*********\n")
    if u'loggerText' in item:
        for row in item[u'loggerText']:
            fh.write(row[:-1] + " , " + "receiveTimeStamp=" + str(item[u'receiveTimeStamp']) + '\n' )
        fh.write("*********\n")    



        
fh.close()    






        
        

     #if u'loggerText' in item:
        #for row in item[u'loggerText']:
            #data = ""
            #nameValuePairs = row.split(',')
            #for nvp in nameValuePairs:
                #name,value = nvp.split('=')
                #if name == 't':
                    #data += name + "=" +str(item[u'receiveTimeStamp']) + ","
                #else:
                    #data += name + "=" + value + ","
            #fh.write(data[:-1])
        #fh.write("*********\n")





    #if u'loggerText' in item:
             #for row in item[u'loggerText']:
                 #data = ""
                 #nameValuePairs = row.split(',')
                 #for nvp in nameValuePairs:
                     #name,value = nvp.split('=')
                     #if name == 't':
                         #data += name + "=" +str(item[u'receiveTimeStamp']) + ","
                     #elif name == ' byte331T':
                         #data += name + "=" +str(item[u'receiveTimeStamp']) + '\n'
                     #elif name == ' byte341T':
                         #data += name + "=" +str(item[u'receiveTimeStamp']) + ","
                     #elif name == ' iteration':
                         #data += name + "=" +str(item[u'receiveTimeStamp']) + '\n'
                     #else:
                         #data += name + "=" + value + ","
                         #if value[-1] == '\n': data = data[:-1]
                 #fh.write(data)
             #fh.write("*********\n")
