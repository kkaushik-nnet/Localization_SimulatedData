#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Splits a recording into border-following and free-mowing phase. Outputs a text file containing info. about
# (time_of_image,time_of_closest_position_estimate,time_dist,filename,x,y) for each phase.
#
# Copyright (C)
# Honda Research Institute Europe GmbH
# Carl-Legien-Str. 30
# 63073 Offenbach/Main
# Germany
#
# UNPUBLISHED PROPRIETARY MATERIAL.
# ALL RIGHTS RESERVED.
#
#


import pylab
import odometryIntegration
import rotMatch
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parseLog(fh):
    imageNameTime = list()
    imageNameTimeBrio = list()
    odo = list()
    odoMessageList = list()
    timeStamptList = list()

    lastT = None
    t0 = 0
    lastOdo = dict()
    isCharging = None
    isWorking = None
    lastbyte341T = 0.
    last_traction_motor_revo_left = 0.
    last_motorState = 0
    last_traction_motor_revo_right = 0

    lineCnt1 = 0
    for line in fh.readlines():
        lineCnt1 += 1
    fh.seek(0)

    lineCnt2 = 0
    for line in fh.readlines():
        lineCnt2 += 1
        if lineCnt2 == lineCnt1: break  # ignore last line, it may be truncated
        if line[0] == '#' or line[0] == '*': continue
        line = line[:-1]

        nameValuePairs = line.split(',')
        # print nameValuePairs

        for nvp in nameValuePairs:
            name, value = nvp.split('=')
            if name[0] == ' ': name = name[1:]
            if name == 'receiveTimeStampKodak': 
                lastT = value
            if name == 'receiveTimeStampBrio':
                lastT = value
            if name == 't':
                lastT = value
                # if t0 == 0: t0 = float(value)
                t0 = 0
            if name in ['saveName', 'yao_pose_x', 'yao_pose_y', 'yao_pose_yaw', 'chargeState', 'workState', 'byte331T',
                        'traction_motor_revo_left', 'traction_motor_revo_right', 'motorState', 'byte341T',
                        'yaw_analog_data', 'frameNameKodak', 'receiveTimeStamp', 'frameNameBrio']:
                # print '*%s*'%name, value
                if lastT is not None:
                    #if name == 'saveName': imageNameTime.append([value, float(lastT) - t0])
                    if name == 'frameNameKodak': imageNameTime.append([value, float(lastT) - t0])
                    if name == 'frameNameBrio': imageNameTimeBrio.append([value, float(lastT) - t0])
                    if name == 'receiveTimeStamp': timeStamptList.append([float(lastT),float(value)])
                    if name in ['yao_pose_x', 'yao_pose_y', 'yao_pose_yaw']:
                        lastOdo[name] = value
                    if name == 'yao_pose_yaw':
                        odo.append(
                            [float(lastT) - t0, lastOdo['yao_pose_x'], lastOdo['yao_pose_y'], lastOdo['yao_pose_yaw'],
                             isCharging, isWorking])
                    if name == 'chargeState':
                        isCharging = float(value)
                    if name == 'workState':
                        isWorking = float(value)
                    if name == 'traction_motor_revo_left': last_traction_motor_revo_left = value
                    if name == 'motorState': last_motorState = value
                    if name == 'traction_motor_revo_right': last_traction_motor_revo_right = value
                    if name == 'byte331T':
                        # got a 331 message, and due to logging order all relevant infos are up to date
                        odoMessageList.append([331, float(value), float(last_traction_motor_revo_left),
                                               float(last_traction_motor_revo_right), int(last_motorState)])
                    if name == 'byte341T': lastbyte341T = value
                    if name == 'yaw_analog_data':
                        # got a 341 message, and due to logging order all relevant infos are up to date
                        odoMessageList.append([341, float(lastbyte341T), float(value)])
    # NOTE: data from each group comes in sync.
    N = len(odo)
    ar = np.zeros([N, 6])
    for n in range(N):
        ar[n, :] = odo[n]
    return imageNameTime, ar, odoMessageList, timeStamptList, imageNameTimeBrio


def matchImageRotation(imgName1, imgName2, minPhi=-4, maxPhi=4, dPhi=0.1):
    # rotate two omni images and find closest match, phi in degrees
    import matplotlib.pyplot as plt
    import scipy
    from scipy import ndimage as ndi
    from skimage import feature

    img1 = (scipy.misc.imread(imgName1).astype('float')).mean(axis=2)[::8, ::8] / 255.
    img2 = (scipy.misc.imread(imgName2).astype('float')).mean(axis=2)[::8, ::8] / 255.

    sim = list()
    phis = list()
    bestImg = None
    bestVal = 99999999999999999
    for phi in np.arange(minPhi, maxPhi, dPhi):
        # pylab.subplot(6,6,cnt+1)
        # cnt+=1
        out = ndi.rotate(img2, phi, mode='constant', reshape=False)
        diff = abs(img1 - out).sum()
        if diff < bestVal:
            bestVal = diff
            bestImg = out
            bestPhi = phi
        sim.append(diff)
        phis.append(phi)

    pylab.figure()
    pylab.plot(phis, sim)
    pylab.title("Img similarity under rotation")
    pylab.figure()
    pylab.imshow(abs(img1 - bestImg))
    pylab.title("Img diff under best rotation")
    pylab.colorbar()

    return bestPhi, bestVal


def parseChargeWorkPhases(odo):
    chargingTimes = list()
    workTimes = list()
    borderFollowTime = list()

    # print 'parseChargeWorkPhases called with message len', len(odo)
    charging = odo[0, 4]
    t0 = odo[0, 0]
    for idx in range(odo.shape[0]):
        elem = odo[idx, :]
        if elem[5] == 1. and len(borderFollowTime) == 0:
            borderFollowTime.append(elem[0])
        if elem[5] == 0. and len(borderFollowTime) == 1:
            borderFollowTime.append(elem[0])
        if elem[4] == charging: continue  # charge state not changed
        t1 = elem[0]
        if charging:
            chargingTimes.append([t0, t1])
        else:
            workTimes.append([t0, t1])
        charging = elem[4]
        t0 = elem[0]
    t1 = elem[0]
    if charging:
        chargingTimes.append([t0, t1])
    else:
        workTimes.append([t0, t1])

    print "charging times:"
    for elem in chargingTimes: print "   from %.1f to %.1f (%.1fs)" % (elem[0], elem[1], elem[1] - elem[0])
    print
    print "work times:"
    for elem in workTimes: print "   from %.1f to %.1f (%.1fs)" % (elem[0], elem[1], elem[1] - elem[0])
    print
    print "border follow time:"
    if len(borderFollowTime) == 2:
        print "   from %.1f to %.1f (%.1fs)" % (
            borderFollowTime[0], borderFollowTime[1], borderFollowTime[1] - borderFollowTime[0])
    else:
        print "Warning: No border following time extracted!"
    return chargingTimes, workTimes, borderFollowTime


def computeDriftSpeed(timesLst, odoMessageList, titleStr=''):
    # estimate yaw drift speed while in base station
    timesAndDrift = list()
    # pylab.figure()  pylab.title(titleStr)
    cnt = 1
    for ct in timesLst:
        # pylab.subplot(len(timesLst),1, cnt)
        cnt += 1
        dt = ct[1] - ct[0]
        yao = OdometryIntegration.YawAssistedWheelOdometry()
        driftplot = list()
        lastYaw = 0.
        for msg in odoMessageList:
            t = msg[1]
            if t < ct[0] or t > ct[1]:
                # not our period
                continue
            if msg[0] == 331:
                yao.updateRevo(msg[1], msg[2], msg[3], msg[4])
            elif msg[0] == 341:
                yao.updateYaw(msg[1], msg[2])
                driftplot.append(yao.yaw - lastYaw)
                lastYaw = yao.yaw
            else:
                raise Exception("Unknown message: " + str(msg))
        # pylab.plot(driftplot) # pylab.title(titleStr)
        yawEnd = yao.yaw
        driftSpeed = yawEnd / dt
        print "Period from %.1f to %.1f (len %.1f) had total drift of %.6f and drift speed of %.6f/s" % (
            ct[0], ct[1], dt, yawEnd, driftSpeed)
        timesAndDrift.append([ct[0], ct[1], driftSpeed])
    return timesAndDrift


def loadMap(fn, rot=0.):
    fh = open(fn, 'r')
    data = list()
    for line in fh.readlines():
        #      160223-101           6.777       -7.584       -0.022
        pts = line.split()
        data.append([float(pts[1]), float(pts[2])])
    ar = np.array(data)
    rotMat = np.matrix(np.eye(2))
    rotMat[0, 0] = rotMat[1, 1] = np.cos(rot)
    rotMat[1, 0] = np.sin(rot)
    rotMat[0, 1] = -np.sin(rot)
    ar = ar * rotMat
    return ar


def loadMapIndoor(fn, rot=0.):
    fh = pd.read_csv(fn, header=None, skiprows=[0])
    data = [fh[1].copy(), fh[2].copy()]
    data = map(list, zip(*data))
    ar = np.array(data)
    rotMat = np.matrix(np.eye(2))
    rotMat[0, 0] = rotMat[1, 1] = np.cos(rot)
    rotMat[1, 0] = np.sin(rot)
    rotMat[0, 1] = -np.sin(rot)
    ar = ar * rotMat
    return ar


def findClosestPos(t, workPeriod):
    # check all time points in work period and find closest match
    dCoord = workPeriod[:, 0]
    tDist = abs(dCoord - t)
    bestIdx = np.argmin(tDist)
    bestDist = tDist[bestIdx]
    x, y, phi = workPeriod[bestIdx, 1:4]
    return x, y, phi, bestDist, workPeriod[bestIdx, 0]


def writeImageTime(workPeriod, imageNameTime, inDir, workPeriodCnt,name):
    # write coordinates csv file with estimated positions of images
    fh = open("%s/coordinates_wp%d_%s.txt" % (inDir, workPeriodCnt,name), 'wt')
    fh.write("#time_of_image,time_of_closest_position_estimate,time_dist,filename,x,y\n")
    maxErr = 0.
    picturePosArr = np.zeros([len(imageNameTime), 2])
    picPosLst = list()
    cnt = 0
    for fn, t in imageNameTime:
        if workPeriod[0, 0] <= t <= workPeriod[-1, 0]:
            x, y, phi, err, t_capture = findClosestPos(t, workPeriod)
            imgName = os.path.basename(fn)
            # print "%.4f,%.4f,%s,%.4f,%.4f,%.4f" %(t,t_capture,imgName,err,x,y)
            fh.write("%.4f,%.4f,%.4f,%s,%.4f,%.4f\n" % (t, t_capture, err, imgName, x, y))
            maxErr = max(maxErr, err)
            picturePosArr[cnt, :] = x, y
            picPosLst.append([imgName, x, y, phi])
            cnt += 1
    print "   max time err:", maxErr
    print "saved coordinate files in", inDir
    dPos = ((picturePosArr[:-1, :] - picturePosArr[1:, :]) ** 2).sum(axis=1) ** 0.5
    print "average / median step width between images:", dPos.mean(), np.median(dPos)
    pylab.figure(50 + workPeriodCnt)
    pylab.hist(dPos, bins=100)
    pylab.title("histogram of travel distances between images")
    pylab.savefig("%s/workPeriodSpeedHisto_%d.png" % (inDir, workPeriodCnt))
    travelledDist = dPos.sum()
    print "total travelled distance: %.4fm" % travelledDist
    return picturePosArr, travelledDist, picPosLst


def rotateTrack(wt, angle):
    xy = wt[:, 1:3]
    rotMat = np.matrix(np.zeros([2, 2]))
    rotMat[0, 0] = rotMat[1, 1] = np.cos(angle)
    rotMat[1, 0] = -np.sin(angle)
    rotMat[0, 1] = np.sin(angle)
    wt[:, 1:3] = xy * rotMat


def plausibilityCheck(picPosLst, inDir, minBSDist=0.5, matchDist=0.05, minIdxDist=10):
    # go through list of [filenames, x, y, phi] and check if images with corresponding positions have relative
    # orientation es expected by phi
    # TODO: unfinished code 23 May 2017

    nTotalMatches = 0
    lastMatch_i = -1000
    for i in range(len(picPosLst)):
        if i - lastMatch_i < minIdxDist:
            # we just matched sth, continue quickly so we dont match hundreds
            continue
        imgName0, x0, y0, phi0 = picPosLst[i]
        if (x0 ** 2 + y0 ** 2) ** 0.5 < minBSDist:
            # too close to base station
            continue
        lastMatch_j = -1000
        for j in range(i + 100, len(picPosLst)):
            if j - lastMatch_j < minIdxDist:
                # we just matched sth, continue quickly so we dont match hundreds
                continue
            imgName1, x1, y1, phi1 = picPosLst[j]
            dst = ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
            if dst < matchDist:
                print "match between index (%d,%d): dist is %.4fm, pos:(%.2f,%.2f) vs (%.2f,%.2f), %s, %s" % (
                    i, j, dst, x0, y0, x1, y1, imgName0, imgName0)
                lastMatch_i = i
                lastMatch_j = j
                nTotalMatches += 1

                bestPhi, bestVal = matchImageRotation("%s/Kodak/sub0/%s" % (inDir, imgName0),
                                                      "%s/Kodak/sub0/%s" % (inDir, imgName1), minPhi=-180, maxPhi=180,
                                                      dPhi=0.1)
                print "yaws according to odometry: %.2f, %.2f, dyaw=%.2f" % (phi0, phi1, xxx)
                pylab.show()

    print "total of %d matches" % nTotalMatches


def translateCameraPos(track, camOffset=.21):
    # compute camera position, which is offset to odometry position by .21m
    # track: t, x, y, yaw
    trackOut = track.copy()
    for i in range(len(track)):
        yaw = track[i, 3]
        #print "Step %d, yaw=%.2f, pos=(%.1f,%.1f), pos update=(%.2f, %.2f)" % (
        #    i, yaw, track[i, 1], track[i, 2], np.cos(yaw) * camOffset, np.sin(yaw) * camOffset)
        trackOut[i, 1] += np.cos(yaw) * camOffset
        trackOut[i, 2] += np.sin(yaw) * camOffset
    return trackOut


if False:
    # example fragment: compute relative rotations of two images at (hopefully) the same position
    inDir = '/hri/localdisk/franzius/slData/sub1/'
    bestPhi, bestVal = matchImageRotation('/hri/localdisk/franzius/slData/sub4/Kodak/sub0/omni001489-R.jpg',
                                          '/hri/localdisk/franzius/slData/sub4/Kodak/sub0/omni018487-R.jpg', minPhi=-4,
                                          maxPhi=4, dPhi=0.1)
    print bestPhi, bestVal
    bestPhi, bestVal = matchImageRotation('/hri/localdisk/franzius/slData/sub4/Kodak/sub0/omni001489-R.jpg',
                                          '/hri/localdisk/franzius/slData/sub4/Kodak/sub0/omni018487-R.jpg',
                                          minPhi=bestPhi - 0.1, maxPhi=bestPhi + 0.1, dPhi=0.005)

    print bestPhi, bestVal
    pylab.show()

if __name__ == "__main__":
    #if len(sys.argv) < 2:
    #    raise Exception("call with location of recording (e.g. /hri/localdisk/franzius/slData/sub1/)")

    inDir = './'#sys.argv[1]
    #if not os.path.exists(inDir):
        #raise Exception("directory %s does not exist" % inDir)
    #if not os.path.exists('%s/log.txt' % inDir):
        #raise Exception("logfile %s/log.txt does not exist" % inDir)

    fh = open('log.txt', 'r')

    # if not os.path.exists('%s/map.PKT'%inDir):
    #  raise Exception ("insert a link in each recording subdir,
    # e.g. map.PKT -> /home/franzius/SelfLocalization/MapData/O4_garden/160316_16703_kleinerGarten_lokal.PKT")

    # For outdoor recordings, small and design garden
    mapPoints = loadMap('160316_16703_kleinerGarten_lokal.PKT') #loadMap') #('map.PKT')	# look for ground truth map data for visualization (and plausibilization)

    #For indoor area
    # mapPoints = loadMapIndoor('indoorMap.csv')  # 'indoorMap.csv' # ('%s/indoorMap.csv'%inDir) 
    # Only for indoor map
    #mapPoints[:, 0] -= 0.0030
    #mapPoints[:, 1] -= 0.4400

    #Plot the reference map
    #plt.plot(mapPoints[:, 0], mapPoints[:, 1], 'r-')
    #plt.show()

    imageNameTime, odoArr, odoMessageList , tsList, imageNameTimeBrio = parseLog(fh)
    arr = np.array(tsList)
    print imageNameTimeBrio
    #print imageNameTime[0]
    #print odoArr[0,:]
    #print odoMessageList[0]
    #print tsList[0]

    #if len(odoMessageList) == 0:
        #raise Exception("no odometry information in log.txt")

    ##  split whole recording in work and charge periods
    ##print odoMessageList
    chargingTimes, workTimes, borderFollowTime = parseChargeWorkPhases(odoArr)
    ##  core part: compute odometry for each work period
    

    workTracks, borderFollowMask = odometryIntegration.computeMultiTrack(odoMessageList, chargingTimes, workTimes,
                                                                        borderFollowTime, mapPoints=mapPoints)
    
    print len(workTracks)
    print workTracks[0][0,0],workTracks[0][-1,0], workTracks[0][-1,0] - workTracks[0][0,0] 

    
    
    pylab.figure(99)
    pylab.plot(workTracks[0][:,1],workTracks[0][:,2],'r.')
    l2, = pylab.plot(workTracks[0][borderFollowMask == False, 1], workTracks[0][borderFollowMask == False, 2], 'rx')
    l1, = pylab.plot(workTracks[0][borderFollowMask, 1], workTracks[0][borderFollowMask, 2], 'b-')
    pylab.legend([l1, l2], ['track', 'return part'])


    fitMap, phi, err = rotMatch.fit(workTracks[0][borderFollowMask, 1:3],
                                    mapPoints)  # rotate GT map to best fit to found odometry
    #workTracks[0]= workTracks[0][borderFollowMask]
    ##print len(workTracks)
  
    ### _new_ ##
    for j in range(len(workTracks)):
        for k in range(len(workTracks[j])):
            search = workTracks[j][k][0]
            tDist = abs(search - arr[:,0])
            bestIdx = np.argmin(tDist)
            workTracks[j][k][0] = arr[bestIdx,1] 
    ############
    print workTracks[0][0,0],workTracks[0][-1,0], workTracks[0][-1,0] - workTracks[0][0,0] 
    for cnt in range(len(workTracks)):  
        # for each work period
        camOffsetTrack = translateCameraPos(workTracks[cnt])
        rotateTrack(workTracks[cnt] , -phi)  # rotate track to match map
        rotateTrack(camOffsetTrack, -phi)



        ##  write csv file about positions and timestamps of recorded images
        picPosArr, travelledDist, picPosLst = writeImageTime(camOffsetTrack, imageNameTime, inDir, cnt,"Kodak")
        #picPosArr, travelledDist, picPosLst = writeImageTime(camOffsetTrack, imageNameTimeBrio, inDir, cnt,"Brio")
        
        ##  plausibilityCheck(picPosLst, inDir)
        ##  optional check if views really match where we calculate that they are at the same positions
        ##   print picPosArr.shape
 
        pylab.figure(100 + cnt)  # visualization

        pylab.plot(workTracks[cnt][:, 1], workTracks[cnt][:, 2], 'r.')
        #pylab.plot(camOffsetTrack[:, 1], camOffsetTrack[:, 2], 'k.')

        ##pylab.plot(fitMap[:,0], fitMap[:,1])
        ##pylab.plot(mapPoints[:, 0], mapPoints[:, 1])

        if cnt == 0:
            pylab.title(
               "WP%d with best map rotation (phi=%.2f, error=%.1f), len=%.2fm" % (cnt, phi, err, travelledDist))
        else:
            pylab.title("WP%d, len=%.2fm" % (cnt, travelledDist))
        pylab.figure(100 + cnt)
        pylab.plot(picPosArr[:, 0], picPosArr[:, 1], '.')
        pylab.axis('equal')

        ## ax.legend(loc = 'lower right',['workTrack', 'camOffsetTrack', 'mapPoints', 'picturePoints'])

        pylab.savefig("%s/workPeriodMap_%d.png" % (inDir, cnt))

    pylab.show()
