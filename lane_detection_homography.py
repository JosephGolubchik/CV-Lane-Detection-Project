import cv2
import numpy as np
import canny
import hough

refPt = []

def main():
    global refPt
    vid_path = 'vids/test7.mp4'
    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", add_point)
    cap = cv2.VideoCapture(vid_path)
    ret, frame = cap.read()
    out = cv2.VideoWriter('res1_opencv.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))
    clone = frame.copy()
    draw_ref = True
    while True:
        # display the image and wait for a keypress
        cv2.imshow("frame", frame)
        out.write(frame)
        if draw_ref:
            if len(refPt) == 1:
                cv2.circle(frame, tuple(refPt[0]), 0, (0, 255, 0), 2)
            if len(refPt) > 1:
                frame = clone.copy()
                cv2.polylines(frame, [np.array(refPt)], True, (0, 255, 0))
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the points
        if key == ord("r"):
            frame = clone.copy()
            refPt = []
            draw_ref = True
        # if 't' is pressed, apply homography and get preview
        if key == ord("t"):
            print(refPt)
            frame = perspective_transform(clone.copy(),  np.float32(refPt),  np.float32([[0,0],[0,720],[1280,720],[1280,0]]))
            draw_ref = False
        # if the 'c' key is pressed, apply homography and continue
        if len(refPt) > 3:
            if key == ord("c"):
                break
        if key == ord("q"):
            exit()
    cap.release()
    cap = cv2.VideoCapture(vid_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
                frame = lane_detection(frame)
                if len(frame.shape) < 3:
                    frame = cv2.merge((frame, frame, frame))
                cv2.imshow('frame', frame)
                out.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            cap.release()
            out.release()
    cv2.destroyAllWindows()


def add_point(event, x, y, flags, param):
    # grab references to the global variables
    global refPt

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONUP:
        if len(refPt) < 4:
            refPt.append([x, y])


def perspective_transform(img, src, dst, inv=False):
    """
    Execute perspective transform
    """
    img_size = (img.shape[1], img.shape[0])

    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)

    if inv is False:
        res = cv2.warpPerspective(img, m, img_size, flags=cv2.INTER_LINEAR)
    else:
        res = cv2.warpPerspective(img, m_inv, img_size, flags=cv2.INTER_LINEAR)

    return res


def lane_detection(img):
    """Detects lanes in a given image using OpenCV functions"""
    persp = perspective_transform(img, np.float32(refPt), np.float32([[0, 0], [0, 720], [1280, 720], [1280, 0]]))
    # Detect yellow lane lines by keeping only yellowish pixels
    yellow = only_yellow(persp)
    # Detect white lane lines by keeping only whiteish pixels
    white = only_white(persp)
    # Combine the detected yellow and white lane lines
    ynw = np.zeros(white.shape)
    w_ind = np.where(yellow > 0)
    y_ind = np.where(white > 0)
    ynw[w_ind] = 1
    ynw[y_ind] = 1
    lines = perspective_transform(sliding_window_lines(ynw)[2], np.float32(refPt), np.float32([[0, 0], [0, 720], [1280, 720], [1280, 0]]), True)
    ind = np.where(lines != 0)
    lines_img = img.copy()
    lines_img[ind] = 0.6*lines[ind] + 0.4*lines_img[ind]

    return lines_img

def sliding_window_lines(img):
    # Take a histogram of the bottom half of the image
    bottom_half_y = img.shape[0] / 2
    histogram = np.sum(img[int(img.shape[0] / 2):, :], axis=0)
    # Create an output image to draw on and  visualize the result
    # out_img = np.dstack((img, img, img)) * 255
    out_img = np.zeros((img.shape[0], img.shape[1], 3))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(img.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # # Draw the windows on the visualization image
        # cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        # cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                    nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                    nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if len(leftx) > 0 and len(lefty) > 0 and len(rightx) > 0 and len(righty) > 0:
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Generate black image and colour lane lines
        # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Draw polyline on image
        right = np.asarray(tuple(zip(right_fitx, ploty)), np.int32)
        left = np.asarray(tuple(zip(left_fitx, ploty)), np.int32)

        cv2.polylines(out_img, [right], False, (255, 255, 0), thickness=5)
        cv2.polylines(out_img, [left], False, (255, 255, 0), thickness=5)
        right = np.vstack((left, right[::-1]))
        cv2.fillPoly(out_img, [right], (255,255,0))

    return left_lane_inds, right_lane_inds, out_img

def draw_lane_rect(img, best_two):
    if best_two == [[]]:
        return img
    points = []
    for line in best_two:
        a = np.cos(line[0][1])
        b = np.sin(line[0][1])
        x0 = a * line[0][0]
        y0 = b * line[0][0]
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 4000 * (-b))
        y2 = int(y0 - 4000 * a)
        m = (y2 - y1) / (x2 - x1)
        b = y1 - (m * x1)
        points.append((m, b))
    if len(points) >= 2:
        p1 = [int((img.shape[0] - points[0][1]) // points[0][0]), int(img.shape[0])]
        p2 = [int((img.shape[0] - points[1][1]) // points[1][0]), int(img.shape[0])]
        xc = int((points[1][1] - points[0][1]) // (points[0][0] - points[1][0] + 0.000001))
        yc = int(points[0][0] * xc + points[0][1])
        p3 = [xc, yc]
        pts = np.array([p1, p2, p3])
        road_mask = img.copy()
        cv2.drawContours(road_mask, [pts], 0, (0, 255, 0, 100), -1)
        road_mask[0:yc + int(img.shape[0] * 0.04), :] = img[0:yc + int(img.shape[0] * 0.04), :]
        img_lines = img * 0.7 + road_mask * 0.3
        img_lines = cv2.normalize(src=img_lines, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                  dtype=cv2.CV_8UC1)
        return img_lines


def adjust_gamma(img, gamma=1.0):
    """Apply gamma correction"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def brighten_dark_frame(img):
    """Keeps all frames at about the same brightness,
    to make sure lane lines are detected in dark frames"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avg_val = np.mean(hsv[img.shape[0] // 2:, :, 2])
    if avg_val < 60:
        v = adjust_gamma(hsv[:,:,2], 1)
        hsv = cv2.merge((hsv[:,:,0], hsv[:,:,1], v))
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def only_white(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hls_white_bin = np.zeros_like(hsv[:, :, 0])
    img_hls_white_bin[((hsv[:, :, 0] >= 0) & (hsv[:, :, 0] <= 255))
                      & ((hsv[:, :, 1] >= 0) & (hsv[:, :, 1] <= 20 / 100 * 255))
                      & ((hsv[:, :, 2] >= 60 / 100 * 255) & (hsv[:, :, 2] <= 255))
                      ] = 1

    img_hls_white_bin = cv2.normalize(src=img_hls_white_bin, dst=None, alpha=0, beta=255,
                                      norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return img_hls_white_bin


def only_yellow(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    img_lab_yellow_bin = np.zeros_like(lab[:, :, 0])
    img_lab_yellow_bin[(lab[:, :, 2] >= 160)] = 1

    img_lab_yellow_bin = cv2.normalize(src=img_lab_yellow_bin, dst=None, alpha=0, beta=255,
                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return img_lab_yellow_bin


def apply_mask(img, mask_path):
    if len(img.shape) > 2:
        mask = cv2.imread(mask_path)
        ret, mask = cv2.threshold(mask, 180, 255, cv2.THRESH_BINARY)
        res = mask * img / 255
        res = cv2.normalize(src=res, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    else:
        mask = cv2.imread(mask_path, 0)
        ret, mask = cv2.threshold(mask, 180, 255, cv2.THRESH_BINARY)
        res = mask * img / 255
        res = cv2.normalize(src=res, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return res


def line_groups(lines):
    lines2 = []
    sorted_lines = [[]]
    if lines is not None and lines != []:
        for line in lines:
            for rho, theta in line:
                lines2.append(tuple((rho, theta)))
        lines2.sort()

        group = 0
        group_thresh = 80

        can_continue = False
        lines = lines2
        i = 0
        while not can_continue:
            sorted_lines[group].append([lines[i]])
            can_continue = True
            prev_line = lines[i]
            i += 1
            if i == len(lines):
                return sorted_lines

        for line in lines[i + 1:]:
            if np.abs(line[0] - prev_line[0]) < group_thresh:
                sorted_lines[group].append([line])
            else:
                group += 1
                sorted_lines.append([])
                sorted_lines[group].append([line])
            prev_line = line
    return sorted_lines


def best_two_avg_lines(avg_lines):
    if len(avg_lines) < 2:
        best_two = [[]]
    else:
        min_diff1 = 1000
        min_diff2 = 2000
        best_line1 = avg_lines[0]
        best_line2 = avg_lines[0]
        for line in avg_lines:
            theta_diff_from_vertical = min(line[0][1], np.pi - line[0][1])
            if theta_diff_from_vertical < min_diff1:
                min_diff2 = min_diff1
                min_diff1 = theta_diff_from_vertical
                best_line1 = line
            elif theta_diff_from_vertical < min_diff2:
                min_diff2 = theta_diff_from_vertical
                best_line2 = line
        best_two = [best_line1, best_line2]
    return best_two


def avg_of_line_groups(lines):
    sorted_lines = line_groups(lines)
    avg_lines = []
    for line_group in sorted_lines:
        if len(line_group) != 0:
            rho_sum = 0
            theta_sin_sum = 0
            theta_cos_sum = 0
            theta_sum = 0
            for line in line_group:
                rho_sum += line[0][0]
                theta_sin_sum += np.sin(line[0][1])
                theta_cos_sum += np.cos(line[0][1])
                theta_sum += line[0][1]
            rho_avg = rho_sum / len(line_group)
            n = len(line_group)
            # rho_avg = line_group[n // 2][0]
            line_group.sort(key=lambda x: x[0][1])
            theta_median = line_group[n // 2][0][1]
            rho_median = line_group[n // 2][0][0]
            # theta_avg = math.atan2((1 / n) * theta_sin_sum, (1 / n) * theta_cos_sum)
            # theta_avg = theta_sum / len(line_group)
            avg_lines.append([[rho_median, theta_median]])
    return avg_lines


def draw_lines(img, lines, color=(0, 0, 255)):
    if len(img.shape) < 3:
        img_lines = cv2.merge((img, img, img))
    else:
        img_lines = img.copy()
    if lines != [[]]:
        points = []
        for line in lines:
            a = np.cos(line[0][1])
            b = np.sin(line[0][1])
            x0 = a * line[0][0]
            y0 = b * line[0][0]
            x1 = int(x0 + 4000 * (-b))
            y1 = int(y0 + 4000 * (a))
            x2 = int(x0 - 4000 * (-b))
            y2 = int(y0 - 4000 * (a))
            m = (y2 - y1) / (x2 - x1 + 0.001)
            b = y1 - (m * x1)
            points.append((m, b))
            cv2.line(img_lines, (x1, y1), (x2, y2), color, 2)
    return img_lines


def hough_lines(img, hor_threshold, ver_threshold):
    lines = cv2.HoughLines(img, 1, np.pi / 180, 20)
    sorted_lines = []
    if lines is not None:
        for line in lines:
            if (np.deg2rad(90 + hor_threshold) < line[0][1] < np.deg2rad(180 - ver_threshold)) or (
                    np.deg2rad(ver_threshold) < line[0][1] < np.deg2rad(90 - hor_threshold)):
                sorted_lines.append(line)
    return sorted_lines


if __name__ == '__main__':
    main()
