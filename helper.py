"""
Description : To provide support functions like reading of image
Created by  : Prof Arbind Kumar Gupta
Dated       : 12 Sep  2022 @11.30
Status      : Working
Changes done: Subsequent changes to minor versions to be recorded here
To Do       : handling of pixel size and max edit distance to be linked
Issues      : None
Version     : 1.00
"""

import os
import numpy as np
import pydicom

root = os.getcwd()
data = root + "/data"

def read_image(path):
    im = pydicom.dcmread(path)
    im = im.pixel_array
    image = np.zeros(shape=(256, 256))
    image[:im.shape[0],:im.shape[1]] = im.copy()
    image = image / (np.max(image))
    im = np.uint8(im * (255 / np.max(im)))
    return im, image


def find_files(extension='.dcm'):
    image_paths = []
    for dirpath, dirnames, files in os.walk(data):
        for name in files:
            if name.endswith(extension):
                image_paths.append(os.path.join(dirpath, name))
    return image_paths

class properties():
    # It stores all the contour related properties used during editing of the contour
    def ctrCentre(s):
        if s.ctr is None: return None
        centre = int(np.mean(s.ctr[:,0])), int(np.mean(s.ctr[:,1]))
        return centre

    def __init__(s, ctr, editDist):
        s.min_radius = 10
        s.radius = 30        # initial radius
        s.max_edit = editDist
        s.oldCtr = 0
        s.ctr = ctr
        s.bPtsEdited = list()  # It will be an array of the size of #contour_points: 0 - not edited, 1, edited by 1, 2 - edited by 2
        s.loc = s.ctrCentre()
