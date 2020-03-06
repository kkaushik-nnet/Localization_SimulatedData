#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Odometry Integration module
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

import time
import numpy
import pylab
from numpy import sin, cos, pi


class YawAssistedWheelOdometry:
    def __init__(self):
        self.x = numpy.float64(0.)
        self.y = numpy.float64(0.)
        self.yaw = numpy.float64(0.)
        self.lastMsgT_yaw = 0.
        self.lastMsgT_rev = 0.
        self.maxDT = 0.

    def updateRevo(self, t, traction_motor_revo_left, traction_motor_revo_right, motorState):
        dt = t - self.lastMsgT_rev
        self.lastMsgT_rev = t
        wr = 0.095  # wheel radius in meter
        wheel_l = 0.400  # wheel axial distance in meter
        rpmScale = 1.0 / 8700  # scaling of rpm values in CAN data
        eps = 0.000001  # threshold for straight mtn

        VL = 2 * pi * wr * traction_motor_revo_left * rpmScale
        VR = 2 * pi * wr * traction_motor_revo_right * rpmScale

        dir_L = motorState & 3  # bits 7,8
        dir_R = (motorState / 4) & 3  # bits 5,6
        # rotation direction (+ forward (2), - backward (1))
        if dir_L == 2:
            VL *= -1  # going back
        elif dir_L == 3:
            VL *= 0.  # brake state: ignore

        if dir_R == 2:
            VR *= -1  # going back
        elif dir_R == 3:
            VR *= 0.  # brake state: ignore

        self.maxDT = max(self.maxDT, dt)
        # print "DT_revo:", dt, "maxDT:", self.maxDT
        w = (VR - VL) / wheel_l  # ori component of movement

        if abs(VL - VR) < eps:
            # straight motion
            v = (VR + VL) / 2
            dx = v * numpy.cos(self.yaw) * dt
            dy = v * numpy.sin(self.yaw) * dt
        else:
            # curved motion
            R = (VR + VL) / (VR - VL) * wheel_l / 2.
            cos_yaw = numpy.cos(self.yaw)
            sin_yaw = numpy.sin(self.yaw)
            dx = R * sin_yaw * numpy.cos(w * dt) + R * cos_yaw * sin(w * dt) - R * sin_yaw
            dy = R * sin_yaw * numpy.sin(w * dt) - R * cos_yaw * cos(w * dt) + R * cos_yaw

        self.x += dx
        self.y += dy

    def updateYaw(self, t, yaw_analog_data):
        dt = t - self.lastMsgT_yaw
        self.lastMsgT_yaw = t
        syaw = (yaw_analog_data - 503.) / 51. + 0.0002  # ATTN: these may vary between Miimos
        syaw = syaw / 180 * pi
        self.yaw -= syaw  # opposite turn direction of yaw and wheel yaw
        # self.yaw = self.yaw % (2*numpy.pi)

    def getCurrentIntegratedPose(self):
        return self.x, self.y, self.yaw


def computeTrack(odometryList, mapPoints=None):
    """Take list of odometry updates [msgType, t, ...]
      MsgType is either 331 for (331, t, traction_motor_revo_left, traction_motor_revo_right, motorState) or 341 for (
      341, t, yaw_analog_data)
      Assume start and end pose (position and yaw angle) is identical (e.g., in base station).
      1. Computes forward yaw-assisted odometry
      2. Computes yaw error and corrects it back in time
      3. Computes position error and corrects it back in time
      4. Returns corrected pose estimates"""
    N = len(odometryList)

    print "computeTrack called with %d messages from t=%d to %d (duration=%d)" % (
        N, odometryList[0][1], odometryList[-1][1], odometryList[-1][1] - odometryList[0][1])

    poseForward = numpy.zeros([N, 4])  # [t, x, y, yaw] after forward integration
    poseOriCorr = numpy.zeros([N, 2])  # [t, dYaw] orientation correction from loop closure
    poseOut = numpy.zeros([N, 4])  # [t, x, y, yaw] after both ori and position loop closure

    yao = YawAssistedWheelOdometry()
    cnt = 0
    for msg in odometryList:
        if msg[0] == 331:
            yao.updateRevo(msg[1], msg[2], msg[3], msg[4])
        elif msg[0] == 341:
            yao.updateYaw(msg[1], msg[2])

        else:
            raise Exception("Unknown message: " + str(msg))
        x, y, yaw = yao.getCurrentIntegratedPose()
        poseForward[cnt, :] = msg[1], x, y, yaw
        cnt += 1

    # loop-close for yaw data (same orientation begin and end)
    yACC = numpy.zeros(N)  # accumulated yaw (orientation)

    # compute accumulated yaw
    for i in range(1, N):
        # compute absolute yaw between current and previous position
        yACC[i] = abs(poseForward[i, 3] - poseForward[i - 1, 3])

        # distribute yaw end-point-error according to acc. yaw
    # compute pure difference
    yEPE = poseForward[-1, 3] - poseForward[0, 3]
    print "   yaw total error / eEP:", yEPE
    dt = odometryList[-1][1] - odometryList[0][1]
    print "   total work time: %.1f" % dt
    print "   yaw drift speed: %.5f" % (yEPE / dt)
    yEPE = (pi - yEPE) % (2 * pi) - pi

    print "start and end yaw: %.2f, %.2f (%.2f mod 2pi)" % (poseForward[0, 3], poseForward[-1, 3],
                                                            poseForward[-1, 3] % (2 * pi))

    # yACC[:] = 1
    yACC /= sum(yACC)

    # distribute correction --> loop closure
    poseOriCorr[:, 0] = poseForward[:, 0]
    poseOriCorr[:, 1] = yACC * yEPE

    yao2 = YawAssistedWheelOdometry()  # for ori-corrected integration
    cnt = 0
    for msg in odometryList:
        if msg[0] == 331:
            yao2.updateRevo(msg[1], msg[2], msg[3], msg[4])
        elif msg[0] == 341:
            yao2.updateYaw(msg[1], msg[2])
        else:
            raise Exception("Unknown message: " + str(msg))

        yao2.yaw += poseOriCorr[cnt, 1]  # add yaw correction signal
        poseOut[cnt, :] = msg[1], yao2.x, yao2.y, yao2.yaw
        cnt += 1

    # now distribute position error back in time
    dx = poseOut[-1, 1] - poseOut[0, 1]
    dy = poseOut[-1, 2] - poseOut[0, 2]
    print "   loop closure spatial error dx=%.4fm, dy=%.4fm" % (dx, dy)

    if False:
        # equally distributed position loop closure
        xCorr = numpy.linspace(0, -dx, N)
        yCorr = numpy.linspace(0, -dy, N)
    else:
        # loop closure weighted by movement speed
        spdx = abs(poseOut[1:, 1] - poseOut[:-1, 1]).astype('float128')
        spdy = abs(poseOut[1:, 2] - poseOut[:-1, 2]).astype('float128')
        tmp = (spdx ** 2 + spdy ** 2) ** 0.5
        # pylab.figure()  pylab.plot(tmp)
        spd = numpy.zeros(poseOut.shape[0], 'float128')
        spd[1:] = tmp  # extend by one time point to fit poseOut

        if spd.sum() < 0.01:
            raise Exception("Work period too small, not driven any distance")

        spd = spd / spd.sum()  # normalize to sum 1
        xCorr = numpy.cumsum(spd) * -dx
        yCorr = numpy.cumsum(spd) * -dy

    poseOut[:, 1] += xCorr
    poseOut[:, 2] += yCorr

    pylab.figure()
    for track in [poseForward, poseOut]:
        pylab.subplot(2, 1, 1)
        track = track.copy()
        track[:, 0] -= track[:, 0].min()  # plot with t0=0
        pylab.plot(track[:, 0], track[:, 1])
        pylab.plot(track[:, 0], track[:, 2])
        pylab.plot(track[:, 0], track[:, 3] % (2 * pi))
        pylab.title('x,y,phi before and after loop closure')

        pylab.subplot(2, 1, 2)
        pylab.plot(track[:, 1], track[:, 2])
        pylab.axis('equal')
        pylab.title('x/y before and after loop closure')

    """
   if mapPoints is not None:
      pylab.plot(mapPoints[:,0], mapPoints[:,1])
   else: 1/0
   """
    print
    return poseOut


def computeMultiTrack(odoMessageList, chargingTimes, workTimes, borderFollowTime, minWorkTime=60, mapPoints=None):
    # compute loop closures for each work segment
    relevantWorkTimes = list()
    for workSegment in workTimes:
        dt = workSegment[1] - workSegment[0]
        if dt > minWorkTime:
            relevantWorkTimes.append(workSegment)
            print "relevant work segment:", workSegment

    results = list()
    for workSegment in relevantWorkTimes:
        thisTrack = list()

        for msg in odoMessageList:
            if workSegment[0] < msg[1] < workSegment[1]:
                thisTrack.append(msg)

        print "message: ", workSegment[0] , msg[1], workSegment[1] 

        print "computing track for %d messages out of a total of %d" % (len(thisTrack), len(odoMessageList))
        res = computeTrack(thisTrack, mapPoints=mapPoints)
        results.append(res)

    # get mask for border following mode of first Track for alignment
    if len(borderFollowTime) == 2:
        borderFollowMask = numpy.logical_and(results[0][:, 0] >= borderFollowTime[0],
                                             results[0][:, 0] <= borderFollowTime[1])
    else:
        borderFollowMask = np.ones(len(results[0])).astype(np.bool)
    return results, borderFollowMask
