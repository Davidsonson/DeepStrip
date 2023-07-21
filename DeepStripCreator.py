import cv2
import numpy as np


def calculate_unit_step(p0, p1):
    return (p1 - p0) / np.linalg.norm(p1 - p0)


def determine_next_point_on_polygon(polygon, current_position, last_index, step_size):
    """

    :param polygon: polygon/contour to move along
    :param current_position: starting point on polygon
    :param last_index: index of last vertex stepped through in the polygon
    :param step_size: how far to travel along the polygon
    :return: new_position: coordinates of next point,
             current_index: index of last vertex stepped through in the polygon,
             stop_this_poly: bool to control whether the supplied contour has been fully stepped through
    """
    current_index = last_index
    dist_to_next_point = np.linalg.norm(current_position - polygon[current_index + 1])
    steps_remaining = step_size
    stop_this_poly = False
    # if the next vertex is close then jump to it
    while dist_to_next_point < steps_remaining:
        current_index += 1
        if current_index + 1 >= len(polygon):
            current_index = 0
            stop_this_poly = True
        steps_remaining -= dist_to_next_point
        current_position = polygon[current_index]
        dist_to_next_point = np.linalg.norm(current_position - polygon[current_index + 1])
    # determine which way to step & step that way
    unit_step = calculate_unit_step(current_position, polygon[current_index + 1])
    new_position = current_position + (steps_remaining * unit_step)
    return new_position, current_index, stop_this_poly


def bilinear_interpolate_many(image, Xs, Ys):
    """

    :param image: np array of the image
    :param Xs: x values to evaluate at
    :param Ys: y values to evaluate at
    :return: image interpolated at supplied positions
    """
    X1 = np.floor(Xs).astype(int)
    X2 = X1 + 1
    Y1 = np.floor(Ys).astype(int)
    Y2 = Y1 + 1
    Q11 = image[X1, Y1]
    Q12 = image[X1, Y2]
    Q21 = image[X2, Y1]
    Q22 = image[X2, Y2]
    result = Q11 * np.expand_dims((X2 - Xs) * (Y2 - Ys), -1)
    result += Q21 * np.expand_dims((Xs - X1) * (Y2 - Ys), -1)
    result += Q12 * np.expand_dims((X2 - Xs) * (Ys - Y1), -1)
    result += Q22 * np.expand_dims((Xs - X1) * (Ys - Y1), -1)
    return result


def generate_strip_row(image, polygon, current_position, last_index: int, out_width: int):
    """
    from a given point, determine direction normal to the contour and generate strip of pixel data in that direction
    :param image: image as np.array
    :param polygon: polygon/contour to move along
    :param current_position: starting point on polygon
    :param last_index: index of last vertex stepped through in the polygon
    :param out_width: how many pixels the output should be
    :return: pixel_values: np.array of the values for the strip,
             pixel_map: np.array mapping the strip to positions in orginal image
    """
    # calculate step along direction of the contour
    unit_step = calculate_unit_step(current_position, polygon[last_index + 1])
    # negative inverse to convert to the normal direction
    norm_step = np.array([-1 * unit_step[1], unit_step[0]])
    # start at one end
    start_pos = current_position - ((out_width / 2) * norm_step)
    end_pos = current_position + ((out_width / 2) * norm_step)
    X = np.linspace(start_pos[0], end_pos[0], out_width)
    Y = np.linspace(start_pos[1], end_pos[1], out_width)
    return bilinear_interpolate_many(image, X, Y), np.stack((X, Y), axis=1)


def create_strip(rgb_image: np.array, mask: np.array, strip_w: int = 80, strip_h: int = 4096, min_contour_len=100):
    """

    :param rgb_image: image to generate strip of
    :param mask: mask to determine where the strip is in the base image
    :param strip_w: how far from the strip to go in each normal direction
    :param strip_h: how long the strip should be
    :param min_contour_len: used to filter contours that are just image noise
    :return: strip_image: np.array of the strip,
             pixel_map: np.array mapping pixels in the strip to their original positions
    """
    padded_rgb_image = cv2.copyMakeBorder(rgb_image, strip_w, strip_w, strip_w, strip_w, cv2.BORDER_REPLICATE)
    padded_mask = cv2.copyMakeBorder(mask, strip_w, strip_w, strip_w, strip_w, cv2.BORDER_CONSTANT, value=0)

    # Find contours in mask
    contours, hierarchy = cv2.findContours(padded_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # filter out the small ones that are just noise
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_contour_len]

    # calculate total arc length of all contours
    perimeter = 0
    for i in range(len(contours)):
        peri = cv2.arcLength(contours[i], True)
        cnt = cv2.approxPolyDP(contours[i], 0.0025 * peri, True)
        perimeter += cv2.arcLength(cnt, True)
        contours[i] = cnt
    # tangential step length along the contours to generate image of desired size
    tan_step_length = perimeter / strip_h

    strip_img = []
    strip2out = []

    # starting at the first point of the first contour
    i = 0
    last_index = 0
    squeezed_c = np.squeeze(contours[i])  # contours have redundant axis for some reason
    squeezed_c = np.append(squeezed_c, [squeezed_c[0]], axis=0)  # put first point at the end to close the polygon
    squeezed_c = np.fliplr(squeezed_c)  # flip because contours are x,y instead of cv2 normal y,x
    current_point = squeezed_c[last_index]

    # get the strip at that point
    j = 1
    strip, pixel_map = generate_strip_row(padded_rgb_image, squeezed_c, current_point, last_index, strip_w)
    strip_img.append(strip)
    strip2out.append(pixel_map)

    # loop until the whole image is created
    while j < strip_h:
        # move along the contour
        current_point, last_index, next_contour = determine_next_point_on_polygon(squeezed_c,
                                                                                  current_point,
                                                                                  last_index,
                                                                                  tan_step_length)
        # get the pixels normal to the contour
        strip, pixel_map = generate_strip_row(padded_rgb_image, squeezed_c, current_point, last_index, strip_w)
        strip_img.append(strip)
        strip2out.append(pixel_map)
        j += 1
        # if the end of this contour was reached then start on the next one
        if next_contour:
            i += 1
            last_index = 0
            squeezed_c = np.squeeze(contours[i])
            squeezed_c = np.append(squeezed_c, [squeezed_c[0]], axis=0)
            squeezed_c = np.fliplr(squeezed_c)
            current_point = squeezed_c[last_index]
    return np.asarray(strip_img), np.asarray(strip2out)


def reconstruct_strip(original_image, strip_image, pixel_map, shift_amt):
    rounded_pixel_map = np.round(pixel_map).astype(int)
    new_img = np.zeros_like(original_image)
    h, w, _ = new_img.shape
    idx = rounded_pixel_map.reshape(-1, 2)
    new_img[np.clip(idx[:, 0] - shift_amt, 0, h-1), np.clip(idx[:, 1] - shift_amt, 0, w-1)] = strip_image.reshape(-1, 3)
    return new_img
