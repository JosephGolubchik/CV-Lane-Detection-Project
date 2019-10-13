import numpy as np
import cv2


def HoughLines(edges, hor_threshold, ver_threshold):
    h,w = edges.shape[:2]
    max_dist = int(np.sqrt(h**2+w**2))
    theta_inc = 1
    hspace = np.zeros((180//theta_inc+1,max_dist*2+1))
    for i in range(h):
        for j in range(w):
            if edges[i,j] == 255:
                for theta in range(0,181,theta_inc):
                    rtheta = np.deg2rad(theta)
                    d = int(j*np.cos(rtheta)+i*np.sin(rtheta))
                    hspace[theta//theta_inc,d+max_dist] += 1
    lines = []
    # thresh = 0.7*hspace.max()
    thresh = 20
    ind = np.where(hspace >= thresh)
    for i in range(len(ind[0])):
        theta = np.deg2rad(ind[0][i])
        d = ind[1][i]-max_dist
        if (np.deg2rad(90 + hor_threshold) < theta < np.deg2rad(180 - ver_threshold)) or (
                np.deg2rad(ver_threshold) < theta < np.deg2rad(90 - hor_threshold)):
            lines.append(np.array([[d, theta]]))
    return lines
