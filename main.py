"""
Description: The main python script to run the Delta Nudge tool
"""

# ----------------------------------------------------------------------------------------------------------------------

from eventHandler import eventHdlr
import os, cv2
import numpy as np
import pydicom

# ----------------------------------------------------------------------------------------------------------------------
def read_image(path):
    path = r'E:\Work\DeltaNudgeTool-py\data\image_04.dcm'
    ds = pydicom.dcmread(path)
    try:
        pSize = ds.pixelSpacing[0]
    except: pSize = 1
    im = ds.pixel_array
    im = np.uint8(im * (255 / np.max(im)))
    try:
        olay = ds.overlay_array(0x6000), ds.overlay_array(0x6002)
        import sys
        if '3.6' in sys.version:
            _, endo, _ = cv2.findContours(olay[0], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            _, epi, _ = cv2.findContours(olay[0], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        else:
            endo, _ = cv2.findContours(olay[0], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            epi, _ = cv2.findContours(olay[0], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        ctrs = endo[0].reshape(-1, endo[0].shape[2]), epi[0].reshape(-1, epi[0].shape[2])
    except:
        ctrs = None
    return im, ctrs, pSize

# ----------------------------------------------------------------------------------------------------------------------
image_files = ['image_04.dcm']
#-----------------------------------------------------------------------------------------------------------------------
i = 0
root = os.getcwd()  # Returns the current directory where the file is being executed
for file in image_files:  #Parsing through all the image files available
    im, ctrs, pSize = read_image(file)
    if ctrs is None:
        print('Image does not have a contour, ignoring file %s', file)
        continue
    for ctype in range(2):  # A single image is iterated twice (for inner contour and outer contour)
        Delta_Nudge_tool = eventHdlr(im, ctrs[ctype], pSize) #display(imm, ctr[ctype], dim, crop, c_save[ctype])
        exit_mode = Delta_Nudge_tool.adjustContour_delta()


