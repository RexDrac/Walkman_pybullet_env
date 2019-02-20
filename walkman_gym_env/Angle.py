import numpy as np
import math

# Return the given angle in radian 
# bounded between -PI and PI

def AngleBound(angle):
    #[-pi,pi]
    return angle - 2.0*math.pi*math.floor((angle + math.pi)/(2.0*math.pi))

# Compute the oriented distance between the two given angle
# in the range -PI/2:PI/2 radian from angleSrc to angleDst
# (Better than doing angleDst-angleSrc)

def AngleDistance(angleSrc, angleDst):
    
    angleSrc = AngleBound(angleSrc)
    angleDst = AngleBound(angleDst)

    max, min = 0, 0
    if (angleSrc > angleDst):
        max = angleSrc
        min = angleDst
    else:
        max = angleDst
        min = angleSrc

    dist1 = max-min
    dist2 = 2.0*math.pi -max + min
 
    if dist1 < dist2:
        if angleSrc > angleDst:
            return -dist1
        else:
            return dist1
        
    else:
        if (angleSrc > angleDst):
            return dist2
        else:
            return -dist2

# Compute a weighted average between the 
# two given angles in radian.
# Returned  angle is between -PI and PI.

def AngleWeightedMean(weight1, angle1, weight2, angle2):

    x1 = math.cos(angle1)
    y1 = math.sin(angle1)
    x2 = math.cos(angle2)
    y2 = math.sin(angle2)

    meanX = weight1*x1 + weight2*x2
    meanY = weight1*y1 + weight2*y2

    return math.atan2(meanY, meanX)
