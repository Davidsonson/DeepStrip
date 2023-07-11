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


def bilinear_interpolate(image, point):
    """
    calculate pixel values at a given point even if that point is not an integer
    :param image: image as np.array
    :param point: position at which to calculate pixel value(s)
    :return: np.array of the calculated values
    """
    x, y = point
    x1 = int(np.floor(x))
    x2 = int(np.ceil(x))
    y1 = int(np.floor(y))
    y2 = int(np.ceil(y))
    if x1 == x and y1 == y:
        return image[x1, y1]
    elif x1 == x:
        Q11 = image[x1, y1]
        Q12 = image[x1, y2]
        denom = (y2 - y1)
        numerator = Q11 * (y2 - y)
        numerator += Q12 * (y - y1)
    elif y1 == y:
        Q11 = image[x1, y1]
        Q12 = image[x2, y1]
        denom = (x2 - x1)
        numerator = Q11 * (x2 - x)
        numerator += Q12 * (x - x1)
    else:
        Q11 = image[x1, y1]
        Q12 = image[x1, y2]
        Q21 = image[x2, y1]
        Q22 = image[x2, y2]
        denom = (x2 - x1) * (y2 - y1)
        numerator = Q11 * (x2 - x) * (y2 - y) + Q21 * (x - x1) * (y2 - y)
        numerator += Q12 * (x2 - x) * (y - y1) + Q22 * (x - x1) * (y - y1)
    return numerator / denom


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
    pixel_values = []
    pixel_map = []
    # start at one end
    eval_pos = current_position - ((out_width / 2) * norm_step)
    # loop through and step until reaching the other end
    for i in range(out_width):
        pixel_map.append(eval_pos.copy())
        pixel_values.append(bilinear_interpolate(image, eval_pos))
        eval_pos += norm_step
    return np.asarray(pixel_values), np.asarray(pixel_map)


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
    for cnt in contours:
        perimeter += cv2.arcLength(cnt, True)
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


x = cv2.imread('original_img.PNG', cv2.IMREAD_UNCHANGED)

rgb = x[:, :, :-1] / 255
alpha = x[:, :, -1]
mask = np.where(alpha > 50, 1, 0).astype(np.uint8)

strip_img, pix_map = create_strip(rgb, mask)

cv2.imwrite("strip.jpg", strip_img)
np.save('pixel_map', pix_map, allow_pickle=True)

