
#!/usr/bin/env python
__author__ = 'Nolan Poulin, nipoulin@wpi.edu'

import numpy as np
from copy import deepcopy
from pprint import pprint
from itertools import product
import operator
import cv2

fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Be sure to use the lower case
arrow_color = (0, 0, 255)

arrows = np.array(["\u2190","\u2191","\u2192","\u2193","o"])

def arrow(img,pt1,pt2,color):
    return cv2.arrowedLine(img,pt1,pt2,color,thickness=4)

def arrowNorth(img,center,length,color):
    pt1 = (int(center[0]+length/2),int(center[1]))
    pt2 = (int(center[0]-length/2),int(center[1]))
    return arrow(img,pt1,pt2,color)

def arrowSouth(img,center,length,color):
    pt1 = (int(center[0]-length/2),int(center[1]))
    pt2 = (int(center[0]+length/2),int(center[1]))
    return arrow(img,pt1,pt2,color)

def arrowWest(img,center,length,color):
    pt1 = (int(center[0]),int(center[1]-length/2))
    pt2 = (int(center[0]),int(center[1]+length/2))
    return arrow(img,pt1,pt2,color)

def arrowEast(img,center,length,color):
    pt1 = (int(center[0]),int(center[1]+length/2))
    pt2 = (int(center[0]),int(center[1]-length/2))
    return arrow(img,pt1,pt2,color)

def drawStay(img,center,length,color):
    return cv2.circle(img,(int(center[0]),int(center[1])),int(length/4),color)

drawArrow = [arrowNorth, arrowSouth, arrowEast, arrowWest, drawStay]


class SolutionVideo(object):

    def __init___(self, video_name, grid_map, grid_num_func, action_list):
        self.grid_map = grid_map
        self.grid_num_func = grid_num_func
        self.action_list = action_list
        map_aspect = grid_map.shape/np.max(grid_map.shape)
        self.out = cv2.VideoWriter(video_name + '.avi', fourcc, 2.0, (512, 512))
        self.large = cv2.resize(map_aspect, (512,512), interpolation=cv2.INTER_NEAREST)

    def render(self, policy):
        Qimg  = np.array((self.large)*255,dtype=np.uint8)
        Qimg = cv2.cvtColor(Qimg, cv2.COLOR_GRAY2BGRA)
        for i,j, in product(range(self.grid_map.shape[0]),range(self.grid_map.shape[1])):
            grid_state = self.grid_num_func([i, j])
            # Only showing DRA-state q1 for now.
            augmented_state = (str(grid_state), 'q1')
            highest_prob_act = max(policy[augmented_state].iteritems(), key=operator.itemgetter(1))[0]
            Qimg = drawArrow[self.action_list.index(highest_prob_act)](Qimg, (32*j+16, 32*i+16), 28, arrow_color)
        Qimg = cv2.cvtColor(Qimg, cv2.COLOR_RGBA2BGRA)
        self.out.write(Qimg)
        #cv2.imshow("occupancy", self.large)
        #cv2.imshow("Q",Qimg)
        #cv2.waitKey(1)

        pass

    # Move this to solver/infer class.
    def get_color(self,state):
        cluster= np.argmax(self.beta[:,state[0],state[1]])
        if cluster == 0:
            return (255*self.beta[cluster,state[0],state[1]],0,0)
        elif cluster == 1:
            return(0,255*self.beta[cluster,state[0],state[1]],255*self.beta[cluster,state[0],state[1]])
        elif cluster == 2:
            return(0,255*self.beta[cluster,state[0],state[1]],0)
        elif cluster == 3:
            return(0,0,255*self.beta[cluster,state[0],state[1]])


# E point when called from Command line.
if __name__=='__main__':
    vid = SolutionVideo('test1')
