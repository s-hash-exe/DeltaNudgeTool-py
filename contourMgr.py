"""
Description : Contour manager class to handle editing of the contour
Created by  : Prof Arbind Kumar Gupta
Dated       : 12 Sep  2022 @11.30
Status      : Working
Changes done: Subsequent changes to minor versions to be recorded here
To Do       : a. handling of pixel size and max edit distance to be linked
              b. ARC_SIZE to be adjusted properly
Issues      : None
Version     : 1.00
"""

import numpy as np
import cv2
import math
DILATE, ERODE = 1, -1

class contourMgr(object):
    def __init__(s, prop, pSize=1):
        s.p = prop         # properties of the contour, defined in helper class
        s.opType = 0 # # it indicates if the contour is to be eroded (circle is outside) or dilated (circle inside)
        s.ARC_SIZE = 5
        s.pixelSize = pSize
        s.direction = list()
        s.INNER_RAD = 0.5  # it is the radius of inner circle when erosion / dilation is done by two pixel (as a % of circle radius)

    # -----------------------------------------------------------------------------------------
    def updateParams(s, params):
        s.p = params

    def ret_comp_cont(s):
        return s.p.ctr

    def update_contour(s, ctr):
        s.p.ctr = ctr

    def findCircle(s,p1, p2, p3):

        """
         Input: Three points (float)
         Desc: To compute the equation of the circle for given points(p1 (x1,y1), p2 (x2,y2), p3 (x3,y3))
         Output: 1. Array containing the co-ordinates of centre of the circle (float)
                 2. Radius of the circle (float)
        """

        if  np.all(np.abs(p1 - p2) <= (2,2)) or \
            np.all(np.abs(p1 - p3) <= (2, 2)) or \
            np.all(np.abs(p3 - p2) <= (2, 2)):
            return (p2, 0)    # Condition where points are collinear

        x1, y1, x2, y2, x3, y3 = p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]

        x12 = x1 - x2
        x13 = x1 - x3

        y12 = y1 - y2
        y13 = y1 - y3

        y31 = y3 - y1
        y21 = y2 - y1

        x31 = x3 - x1
        x21 = x2 - x1

        # x1^2 - x3^2
        sx13 = pow(x1, 2) - pow(x3, 2)

        # y1^2 - y3^2
        sy13 = pow(y1, 2) - pow(y3, 2)

        # x2^2 - x1^2
        sx21 = pow(x2, 2) - pow(x1, 2)

        # y2^2 - y1^2
        sy21 = pow(y2, 2) - pow(y1, 2)

        f = (((sx13) * (x12) + (sy13) *
              (x12) + (sx21) * (x13) +
              (sy21) * (x13)) // (2 *
                                  ((y31) * (x12) - (y21) * (x13))))

        g = (((sx13) * (y12) + (sy13) * (y12) +
              (sx21) * (y13) + (sy21) * (y13)) //
             (2 * ((x31) * (y12) - (x21) * (y13))))

        c = (-pow(x1, 2) - pow(y1, 2) -
             2 * g * x1 - 2 * f * y1)

        # eqn of circle be x^2 + y^2 + 2*g*x + 2*f*y + c = 0
        # where centre is (h = -g, k = -f) and
        # radius r as r^2 = h^2 + k^2 - c
        h = -g
        k = -f
        sqr_of_r = h * h + k * k - c

        # r is the radius
        r = round(math.sqrt(sqr_of_r), 5)
        return (np.array((h, k)), r)


    def distance(s,p1, p2):
        """
           Input: Two points (float)
           Desc:  Calculating the distance between point p1 and p2
           Output: Distance (float)
        """
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


    def findDir(s,pt1, pt2):
        """
           Input: Two points (float)
           Descprocess_im:  Determining the direction of pt2 from pt1
           Output: theta(angle) (float)
        """
        th = math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]) # atan = tan ^ -1
        # ERROR CHECK - what happens if it is 0, 0
        th = th * 180 / math.pi
        th = (360 + th) % 360
        return th


    def findClosestPtOnCtr(s,th,dir_list=None):
        """
          Input: theta(angle) (float)
          Desc:  Finding the closest point on the contour in the direction specified by theta
          Output: Closest point on the contour (float)
        """
        closest, diff = -1, 1000
        dirs = s.direction if dir_list is None else dir_list
        for i, th2 in enumerate(dirs):
            if abs(th - th2) < diff:
                closest, diff = i, abs(th - th2)
        return closest

    # -----------------------------------------------------------------------------------------
    def circleInCtr(s, cirCen, ctr, ctrCen=None):
        """
              Input: 1. centre of the circle (Delta Nudge tool)
                     2. contour
                     3. centre of the contour
              Desc:  Finding the closest point on the contour in the direction specified by theta
              Output: Closest point on the contour (float)
        """

        if ctrCen is None:
            ctrCen = (np.mean(ctr[:, 0]), np.mean(ctr[:, 1]))
            s.direction = [s.findDir(ctrCen, pt) for pt in ctr]  # calculate the direction of all the points on the contour

        # Finding direction of circleCentre from contourCentre
        cirDir = s.findDir(ctrCen, cirCen)
        closestPt = ctr[s.findClosestPtOnCtr(cirDir)]
        if s.distance(closestPt, ctrCen) >= s.distance(cirCen, ctrCen):
            return True
        return False

    # -----------------------------------------------------------------------------------------
    def getPoint(s, dir, len, cen):
        """
              Input: 1. dir -  direction of point in contour
                     2. len - The total length of the curve
                     3. centre of the contour
              Desc: To get the point co-ordinates
              Output: Point co-ordinates(float)
        """
        ptDir = dir * math.pi / 180
        relPt = np.array([len * math.cos(ptDir), len * math.sin(ptDir)])
        return (cen + relPt)


    # -----------------------------------------------------------------------------------------
    def getCirclePts(s,cen, rad, fPt, lPt, numPts):
        """
        :param cen: The centre of the circle
        :param rad: Radius
        :param fPt: First point
        :param lPt: Last point
        :param numPts: No. of points
        :return: Points on the circle
        Desc: To obtain points of the circle
        """

        pts = []
        dir1, dir2 = s.findDir(cen, fPt), s.findDir(cen, lPt)
        if abs(dir1 - dir2) > 180:  # if the points on either side of 0 deg (like 5, 355)
            if dir1 > dir2:     # adjust so that dir2 > dir1 and diff is less than 180
                dir2 += 360
            else:
                dir2 -= 360
        step = (dir2 - dir1) / (numPts + 1)

        th = dir1 + step
        while abs(th - dir2) > abs(step * 1.1):
            pt = s.getPoint((th + 360) % 360, rad, cen)
            pts.append(pt)
            th += step
        return np.array(pts)

    # -----------------------------------------------------------------------------------------
    def adjustPt(s, mctr, cirCen, ctrCen, first, last, opType):
        # opType is 1 for dilation and -1 for erosion
        # It shifts the point in the radial direction by 1 or 2 pixels, depending on distance of the point from circle centre
        ctr = s.p.oldCtr.copy()
        idx = ((first + last) // 2) % len(mctr)

        dist = s.distance(ctr[idx], cirCen)
        delChange = 1 if dist > s.INNER_RAD * s.p.radius else 2
        delChange *= opType * s.p.max_edit * s.pixelSize  #max_edit: Maximum editable distance by delta nudge tool; length of the point from the circle centre is increased by this amount
        if delChange > 2: print('adjust contour: ', delChange)
        maxD = s.distance(ctr[idx], ctrCen) + delChange
        midPt = s.getPoint(s.direction[idx], maxD, ctrCen)

        if last - first < s.ARC_SIZE:
            return
        centre, rad = s.findCircle(s.p.ctr[first], midPt, s.p.ctr[last % len(mctr)])
        pts = s.getCirclePts(centre, rad, ctr[first], ctr[last % len(mctr)], last - first - 1)
        for i, pt in zip(range(first + 1, last), pts):
            if opType == DILATE:
                if s.distance(pt, ctrCen) > s.distance(mctr[i % len(mctr)], ctrCen):
                    mctr[i % len(mctr)] = pt
            else:
                if s.distance(pt, ctrCen) < s.distance(mctr[i % len(mctr)], ctrCen):
                    mctr[i % len(mctr)] = pt
        return

    # -----------------------------------------------------------------------------------------
    def updateContour(s, cirCen, opType):
        if s.p.max_edit == 0:             return
        s.opType = opType
        ctrCen = (np.mean(s.p.ctr[:, 0]), np.mean(s.p.ctr[:, 1]))
        inside = s.circleInCtr(cirCen, s.p.ctr, ctrCen)
        if not (inside and s.opType == DILATE or not inside and s.opType == ERODE):
            return s.p.ctr  # the circle should be outside for erosion and inside for dilation

        first, last = -1, -1  # the first and last point on the contour that lie inside the circle
        dilateD, erodeD = 10000, 0  # min and max distance from "cen", ie  cirCen or ctrCen, depending on opType
        cen = ctrCen if s.opType == ERODE else cirCen

        if s.distance(s.p.ctr[0], cirCen) < s.p.radius:            shift = len(s.p.ctr) // 2
        else:            shift = 0
        """ this is to handle the case when the first ctr pt is inside the arc. In that caase
        the search will start from i+shift in the coming for loop. This is to make sure that 
         the first point is outside of the circle arc. """

        for i in range(len(s.p.ctr)):
            d = s.distance(s.p.ctr[(i + shift) % len(s.p.ctr)], cirCen)
            if d < s.p.radius:  # if the point is inside the circle
                if first == -1:                                 first = i + shift
                if last == -1 or (i + shift - last) <= 2:       last = i + shift
        if first >= len(s.p.ctr):
            first, last = first % len(s.p.ctr), last % len(s.p.ctr)
        if first != -1:
            s.adjustPt(s.p.ctr, cirCen, ctrCen, first, last, s.opType)
        return s.p.ctr

    # -----------------------------------------------------------------------------------------
    def dense(s):
        dists = []
        candidates = []
        for i, pt in enumerate(s.p.ctr):
            if i != len(s.p.ctr) - 1:
                dist = s.distance(s.p.ctr[i], s.p.ctr[i + 1])
            else:
                dist = s.distance(s.p.ctr[i], s.p.ctr[0])
            dists.append(dist)

        cutoff = 10
        offset = 0
        for i, d in enumerate(dists):
            num = int(d / cutoff)
            if num == 0:
                continue
            if i != len(dists) - 1:
                points = [s.p.ctr[i + offset] + ((f + 1) / (num + 1)) * (s.p.ctr[i + 1 + offset] - s.p.ctr[i + offset])
                          for f in range(num)]
            else:
                points = [s.p.ctr[i + offset] + ((f + 1) / (num + 1)) * (s.p.ctr[0] - s.p.ctr[i + offset])
                          for f in range(num)]
            points = [pt.reshape([-1, 2]) for pt in points]
            points = np.concatenate(points, axis=0)
            s.p.ctr = np.insert(s.p.ctr, i + 1 + offset, points, axis=0)
            offset += points.shape[0]

        s.p.bPtsEdited = np.array([0] * len(s.p.ctr), int) # to indicate if a boundary point has already been modified
        return s.p.ctr, s.p.bPtsEdited

    def getCentre(s,ctr):
        # from cv2 import moments
        loc = [0, 0]
        ctr = np.asarray(ctr)
        ctr = ctr.reshape(-1, 2)
        m = cv2.moments(ctr)  # contour moments : set of scalar values that describe the distribution of pixels in an image or a contour
        loc = int(m['m10'] // m['m00']), int(m['m01'] // m['m00'])

        return loc

def mainFunc(contour, mousePoint):

    obj = contourMgr(None)

    # Converting list to numpy array
    oCtr = np.asarray(contour)
    # Finding contour centre
    # ctrCentre = (np.mean(oCtr[:, 0]), np.mean(oCtr[:, 1]))
    ctrCentre = obj.getCentre(contour)
    # Finding mouse direction w.r.t. contour centre
    mouseDir = obj.findDir(ctrCentre,mousePoint)

    # Finding directions of all points on contour
    pointDirs = [obj.findDir(ctrCentre, pt) for pt in oCtr]

    theta = 15
    # Taking first and last point directions in a spread of 15 degree on each side of mouseDir
    firstPointDir = (mouseDir - theta) if ((mouseDir - theta) > 0) else (mouseDir + 360 - theta)
    lastPointDir = (mouseDir + theta) % 360
    print("Directions of first, mouse, last point : {}, {}, {}".format(firstPointDir,mouseDir, lastPointDir))

    # Finding first and last point index
    firstPointIdx = obj.findClosestPtOnCtr(firstPointDir, pointDirs)
    lastPointIdx = obj.findClosestPtOnCtr(lastPointDir, pointDirs)

    firstPoint = oCtr[firstPointIdx]
    lastPoint = oCtr[lastPointIdx]

    return [firstPointIdx, lastPointIdx, ctrCentre]