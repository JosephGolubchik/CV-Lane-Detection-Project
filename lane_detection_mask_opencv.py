import cv2
import numpy as np


def main():
    cap = cv2.VideoCapture('vids/test1.mp4')
    prev_best_two = None
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:

            frame, prev_best_two = lane_detection(frame, prev_best_two)
            cv2.imshow('', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            cap.release()
    cv2.destroyAllWindows()


def lane_detection(img, prev_best_two=None):
    """Detects lanes in a given image"""
    # Detect yellow lane lines by keeping only yellowish pixels
    yellow = only_yellow(img)
    # Detect white lane lines by keeping only whiteish pixels
    white = only_white(img)
    # Combine the detected yellow and white lane lines
    ynw = np.zeros(white.shape)
    w_ind = np.where(yellow > 0)
    y_ind = np.where(white > 0)
    ynw[w_ind] = 1
    ynw[y_ind] = 1
    # Apply a region of interest mask to the detected lane line pixels to get rid of irrelevent information
    masked = apply_mask(ynw, 'img/mask11.png')
    # Apply canny edge detection
    edges = cv2.Canny(masked, 100, 200)
    # Find lines in the edge image using the hough transform
    lines = hough_lines(edges, 15, 15)
    # Group all the found lines into seperate groups and generate an average line for every group
    avg_lines = avg_of_line_groups(lines)
    # Choose the two average lines that are closest to a straight up line, because that means they're closest to the
    # middle of the lane
    best_two = best_two_avg_lines(avg_lines)
    # Final two best lines should depend on previous two best lines to make the final result less jittery
    # and prevent sudden accidental jumps
    prev_importance = 0.8
    if prev_best_two is not None:
        if len(best_two) < 2:
            best_two = prev_best_two
        diff = abs(best_two[0][0][0] - prev_best_two[0][0][0]) + abs(best_two[0][0][1] - prev_best_two[0][0][1]) + \
               abs(best_two[1][0][0] - prev_best_two[1][0][0]) + abs(best_two[1][0][1] - prev_best_two[1][0][1])
        if diff > 80:
            best_two = prev_best_two
        else:
            best_two = [[[(1 - prev_importance) * best_two[0][0][0] + prev_importance * prev_best_two[0][0][0],
                          (1 - prev_importance) * best_two[0][0][1] + prev_importance * prev_best_two[0][0][1]]],
                        [[(1 - prev_importance) * best_two[1][0][0] + prev_importance * prev_best_two[1][0][0],
                          (1 - prev_importance) * best_two[1][0][1] + prev_importance * prev_best_two[1][0][1]]]]
    prev_best_two = best_two
    # Draw the lane in green on top of the frame
    rect_img = draw_lane_rect(img, best_two)
    return rect_img, prev_best_two


def draw_lane_rect(img, best_two):
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
        img = img * 0.7 + road_mask * 0.3
        img_lines = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                  dtype=cv2.CV_8UC1)
        return img_lines


def only_white(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hls_white_bin = np.zeros_like(hsv[:, :, 0])
    img_hls_white_bin[((hsv[:, :, 0] >= 0) & (hsv[:, :, 0] <= 255))
                      & ((hsv[:, :, 1] >= 0) & (hsv[:, :, 1] <= 30 / 100 * 255))
                      & ((hsv[:, :, 2] >= 55 / 100 * 255) & (hsv[:, :, 2] <= 255))
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
        group_thresh = 100

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
