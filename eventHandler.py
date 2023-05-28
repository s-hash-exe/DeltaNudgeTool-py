"""
Description : Event handler class to handle mouse and keyboard press while editing of the contour
Created by  : Prof Arbind Kumar Gupta
Dated       : 12 Sep  2022 @11.30
Status      : Working
Changes done: Subsequent changes to minor versions to be recorded here
To Do       : Saving of contour as part of the dicom image
Issues      : None
Version     : 1.00
"""

import cv2, numpy as np
from helper import properties as cParam
from displayMgr import displayMgr
from contourMgr import contourMgr
from contourMgr import  DILATE, ERODE
INIT_WINSIZE = 35
INNER_RAD = 0.5
RAD_DELTA = 2
SIZE_DELTA = 5      # CHange in size of window in number of pixels (while zooming)
MAX_EDIT_DIST = 1   # maximum edit distance for delta nudge tool

# ---------------------------------------------------------------------------------
class eventHdlr(object):

    def __init__(s, img_, ctr, pSize):
        s.img, s.orImg = img_.copy(), img_      # Maintaining a copy of the image array
        s.bdown, s.doubleClick = False, False
        s.cursorLoc = (0, 0)
        s.setInstructions()
        s.opType = 0 # # it indicates if the contour is to be eroded (circle is outside) or dilated (circle inside)
        s.scale, s.wSize = 1,  0
        s.translate = 0
        s.delta, s.key = SIZE_DELTA, None
        s.p = cParam(ctr, MAX_EDIT_DIST)       # properties of the contour for processing by contourMgr class
        s.pSize = pSize
        s.ctrMgr = None
        s.display = displayMgr(mouseEvent=s.mouseEvent, keyEvent=s.onKeyPress)
        s.status = 'NOT DONE'

    # ----------------------------------------------------------------------------------------------------------------------
    def setInstructions(s):
        s.instructions = {
            1: 'left/right arrow - inc/dec cursor size',
            2: 'up/down arrow - inc/dec nudge value',
            3: '-/+(=) - Zoom out / in',
            4: 'Esc - quit'
            }

    # -----------------------------------------------------------------------------------------
    def mouseEvent(s, event):
        type = str(event).split(':')[0]
        if      type == 'button_press_event':       type = 'onMBPress'
        elif    type == 'button_release_event':     type = 'onRel'
        elif    type == 'motion_notify_event':      type = 'onMove'
        else:   print('Wrong mouse event received: ', type)

        x, y = event.xdata, event.ydata
        if x is None: return
        if type == 'onMBPress':
            if event.dblclick:                          s.doubleClick = True
            s.cursorLoc = (x, y)
            s.bdown = True
        elif type == 'onMove' and s.bdown == True:      s.cursorLoc = (x, y)
        elif type == 'onRel':                           s.bdown = False
        elif type == 'onMove':                          s.cursorLoc = (x, y)
        s.type = type

        s.processInput()

    # -----------------------------------------------------------------------------------------
    def onKeyPress(s, event):
        s.key = k = event.key
        if k is None: return
        s.processInput()

    # -----------------------------------------------------------------------------------------
    def setContour(s, translate, scale):       # Adjusting the contour to account for cropped area. s.scale is
        # the old scaling factor and scale is the new scaling factor
        s.p.ctr[:, 0] = (s.p.ctr[:, 0] // s.scale - translate[0]) * scale
        s.p.ctr[:, 1] = (s.p.ctr[:, 1] // s.scale - translate[1]) * scale
        s.p.ctr, s.p.bPtsEdited = s.ctrMgr.dense()       # make it a close list of points
        s.p.oldCtr = s.p.ctr.copy()
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def process_image(s, translate=0):    # translate = 0 indicates first time call to the function
        if translate == 0:
            translate = s.p.loc[0]-INIT_WINSIZE, s.p.loc[1]-INIT_WINSIZE
            s.wSize = INIT_WINSIZE
        else:
            s.wSize += translate
            translate = -translate, -translate    # x shift and y shift of the origin of new image
        if s.wSize < 0 or s.wSize > min(s.img.shape):
            print('ERROR: window size has become zero')

        s.img = s.orImg[s.p.loc[1]-s.wSize:s.p.loc[1]+s.wSize,s.p.loc[0]-s.wSize:s.p.loc[0]+s.wSize]
        newScale = round(s.orImg.shape[0] / (2*s.wSize))   # so that it is integrer
        dim = 2 * s.wSize * newScale
        s.img = cv2.resize(s.img, dsize=(dim, dim))

        s.setContour(translate, newScale)
        s.scale = newScale
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def draw_cursor(s, cursorLoc, radius):
        s.display.circle(cursorLoc, radius, fill=False, color='red')
        s.display.circle(cursorLoc, INNER_RAD * radius, fill=False, color='red')

    # ----------------------------------------------------------------------------------------------------------------------
    def display_text(s):
        h, w = s.img.shape
        C, fontsize = 30, 8
        pos = (10, h//C)

        for i, line in enumerate(s.instructions.values()):
            s.display.text(pos[0], pos[1] + i * h//C, line, fontsize = fontsize, color='red')

        shift = pos[1] + (len(s.instructions)+1) * h//C

    # ----------------------------------------------------------------------------------------------------------------------
    def saveContour(s, ctr):
        s.p.ctr = ctr
        s.p.ctr = s.p.ctr / s.scale
        s.p.ctr[:, 0] += s.p.loc[1]-s.wSize
        s.p.ctr[:, 1] += s.p.loc[1]-s.wSize
        s.status = 'DONE'
        print('done')

    #---------------------------------------------------------------------------------
    def refreshWin(s):
        s.display.clf()
        s.display.imshow(s.img)
        s.display_text()
        s.display.plot(s.p.ctr[:,0], s.p.ctr[:,1])
        s.draw_cursor(s.cursorLoc, s.p.radius)

        s.display.imshow(None, block=False)     # now display the updated image and contour

    #---------------------------------------------------------------------------------
    def adjustContour_delta(s):
        s.ctrMgr = contourMgr(s.p, s.pSize)
        s.process_image()
        s.ctrMgr.update_contour(s.p.ctr)
        s.display.imshow(s.img),    s.display_text()
        s.display.plot(s.p.ctr[:, 0], s.p.ctr[:, 1], 'r')
        s.display.imshow(None, show=True, block=False)

        while s.status != 'DONE':
            s.display.pause(0.02)   # infinite loop - The main thread has to pause every 20ms for event handlers to work

        s.display.close()
        return True

    #---------------------------------------------------------------------------------
    def processMouseInput(s):
        if s.bdown is False:
            s.opType = 0
            s.p.bPtsEdited *= 0  # reset the list of boundary points that were edited
            s.p.oldCtr, s.p.max_edit = s.p.ctr.copy(), MAX_EDIT_DIST
        if s.bdown == True:
            if s.opType == 0:   # set the value for the first time call
                inside = s.ctrMgr.circleInCtr(s.cursorLoc, s.p.ctr)
                s.opType = DILATE if inside else ERODE
            s.ctrMgr.updateContour(s.cursorLoc, s.opType)
        if s.doubleClick:   # end of editing using draw tool
            s.doubleClick = False
            s.saveContour(s.p.ctr)
            return

    #---------------------------------------------------------------------------------

    def processKeyInput(s, key):
        if key == 'escape':             return 'exit'
        elif key == 'up':               s.p.max_edit += 1 if s.p.max_edit < 10 else 0
        elif key == 'down':             s.p.max_edit -= 1 if s.p.max_edit > 1 else 0
        elif key == '-':                s.process_image(s.delta)
        elif key == '+' or key == '=':  s.process_image(-s.delta)
        elif key == 'right':            s.p.radius += RAD_DELTA
        elif key == 'left':             s.p.radius -= RAD_DELTA if s.p.radius > s.p.min_radius else 0

        s.ctrMgr.updateParams(s.p)
        s.ctrMgr.updateContour(s.cursorLoc, s.opType)

    #---------------------------------------------------------------------------------
    def processInput(s):
        s.processMouseInput()
        #handle key board inputs
        key = s.key
        s.key = None      # reset the key received for next input
        s.processKeyInput(key)
        # Now display the image and plot the contour
        s.refreshWin()
        return None
