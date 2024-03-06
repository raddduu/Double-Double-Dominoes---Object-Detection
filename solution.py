import numpy as np
import os
import cv2 as cv

scores_to_dice_number = {
    0: -1,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 0,
    8: 2,
    9: 5,
    10: 3,
    11: 4,
    12: 6,
    13: 2,
    14: 2,
    15: 0,
    16: 3,
    17: 5,
    18: 4,
    19: 1,
    20: 6,
    21: 2,
    22: 4,
    23: 5,
    24: 5,
    25: 0,
    26: 6,
    27: 3,
    28: 4,
    29: 2,
    30: 0,
    31: 1,
    32: 5,
    33: 1,
    34: 3,
    35: 4,
    36: 4,
    37: 4,
    38: 5,
    39: 0,
    40: 6,
    41: 3,
    42: 5,
    43: 4,
    44: 1,
    45: 3,
    46: 2,
    47: 0,
    48: 0,
    49: 1,
    50: 1,
    51: 2,
    52: 3,
    53: 6,
    54: 3,
    55: 5,
    56: 2,
    57: 1,
    58: 0,
    59: 6,
    60: 6,
    61: 5,
    62: 2,
    63: 1,
    64: 2,
    65: 5,
    66: 0,
    67: 3,
    68: 3,
    69: 5,
    70: 0,
    71: 6,
    72: 1,
    73: 4,
    74: 0,
    75: 6,
    76: 3,
    77: 5,
    78: 1,
    79: 4,
    80: 2,
    81: 6,
    82: 2,
    83: 3,
    84: 1,
    85: 6,
    86: 5,
    87: 6,
    88: 2,
    89: 0,
    90: 4,
    91: 0,
    92: 1,
    93: 6,
    94: 4,
    95: 4,
    96: 1,
    97: 6,
    98: 6,
    99: 3,
    100: 0,
}


coordonates_to_scores = {
    (1, 1): 5,
    (1, 4): 4,
    (1, 8): 3,
    (1, 12): 4,
    (1, 15): 5,
    (2, 3): 3,
    (2, 6): 4,
    (2, 10): 4,
    (2, 13): 3,
    (3, 2): 3,
    (3, 5): 2,
    (3, 11): 2,
    (3, 14): 3,
    (4, 1): 4,
    (4, 4): 3,
    (4, 6): 2,
    (4, 10): 2,
    (4, 12): 3,
    (4, 15): 4,
    (5, 3): 2,
    (5, 5): 1,
    (5, 7): 1,
    (5, 9): 1,
    (5, 11): 1,
    (5, 13): 2,
    (6, 2): 4,
    (6, 4): 2,
    (6, 6): 1,
    (6, 10): 1,
    (6, 12): 2,
    (6, 14): 4,
    (7, 5): 1,
    (7, 11): 1,
    (8, 1): 3,
    (8, 15): 3,
    (9, 5): 1,
    (9, 11): 1,
    (10, 2): 4,
    (10, 4): 2,
    (10, 6): 1,
    (10, 10): 1,
    (10, 12): 2,
    (10, 14): 4,
    (11, 3): 2,
    (11, 5): 1,
    (11, 7): 1,
    (11, 9): 1,
    (11, 11): 1,
    (11, 13): 2,
    (12, 1): 4,
    (12, 4): 3,
    (12, 6): 2,
    (12, 10): 2,
    (12, 12): 3,
    (12, 15): 4,
    (13, 2): 3,
    (13, 5): 2,
    (13, 11): 2,
    (13, 14): 3,
    (14, 3): 3,
    (14, 6): 4,
    (14, 10): 4,
    (14, 13): 3,
    (15, 1): 5,
    (15, 4): 4,
    (15, 8): 3,
    (15, 12): 4,
    (15, 15): 5
}


def find_color_values_using_trackbar(frame):

    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
 
    def nothing(x):
        pass

    cv.namedWindow("Trackbar") 
    cv.createTrackbar("LH", "Trackbar", 0, 255, nothing)
    cv.createTrackbar("LS", "Trackbar", 0, 255, nothing)
    cv.createTrackbar("LV", "Trackbar", 0, 255, nothing)
    cv.createTrackbar("UH", "Trackbar", 255, 255, nothing)
    cv.createTrackbar("US", "Trackbar", 255, 255, nothing)
    cv.createTrackbar("UV", "Trackbar", 255, 255, nothing)
    
    
    while True:

        l_h = cv.getTrackbarPos("LH", "Trackbar")
        l_s = cv.getTrackbarPos("LS", "Trackbar")
        l_v = cv.getTrackbarPos("LV", "Trackbar")
        u_h = cv.getTrackbarPos("UH", "Trackbar")
        u_s = cv.getTrackbarPos("US", "Trackbar")
        u_v = cv.getTrackbarPos("UV", "Trackbar")


        l = np.array([l_h, l_s, l_v])
        u = np.array([u_h, u_s, u_v])
        mask_table_hsv = cv.inRange(frame_hsv, l, u)        

        res = cv.bitwise_and(frame, frame, mask=mask_table_hsv)    
        cv.imshow("Mask", mask_table_hsv)
        cv.imshow("Res", res)

        if cv.waitKey(25) & 0xFF == ord('q'):
                break
    cv.destroyAllWindows()


def show_image(image, mode="same", ratio=0.2, title="Image"):
    if mode == "same":
        cv.imshow(title, image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    elif mode == "resize":
        img = cv.resize(image, (0,0), fx=ratio, fy=ratio)
        cv.imshow(title, img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print("Mode not recognized")


def get_open_image(source_image, kernel_size=2):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    source_image = cv.erode(source_image, kernel, iterations=1)
    source_image = cv.dilate(source_image, kernel, iterations=1)

    return source_image

def get_closed_image(source_image, kernel_size=2):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    source_image = cv.dilate(source_image, kernel, iterations=1)
    source_image = cv.erode(source_image, kernel, iterations=1)

    return source_image


def get_freq(img):
    freq = np.zeros((img.shape[0], img.shape[1]))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][0] != 0 and img[i][j][1] != 0 and img[i][j][2] != 0:
                freq[i][j] = i + j

    return freq


def get_corner(freq, corner="LR"):
    if corner == "LR":
        max_value = freq.max()

        max_values_x = np.where(freq == max_value)[0]

        max_values_y = np.where(freq == max_value)[1]

        middle_value_x = max_values_x[len(max_values_x) // 2]

        middle_value_y = max_values_y[len(max_values_y) // 2]

        middle_value = (middle_value_x, middle_value_y)

        return middle_value
    elif corner == "UL":
        min_value = freq[freq > 0].min()

        min_values_x = np.where(freq == min_value)[0]

        min_values_y = np.where(freq == min_value)[1]

        middle_value_x = min_values_x[len(min_values_x) // 2]

        middle_value_y = min_values_y[len(min_values_y) // 2]

        middle_value = (middle_value_x, middle_value_y)

        return middle_value
    

def add_red_corners(img, first_pixel, last_pixel, size=3):
    red = [0, 0, 255]

    img[first_pixel[0]-(size-2):first_pixel[0]+(size-1), first_pixel[1]-(size-2):first_pixel[1]+(size-1)] = red
    img[last_pixel[0]-(size-2):last_pixel[0]+(size-1), last_pixel[1]-(size-2):last_pixel[1]+(size-1)] = red

    return img


def get_table_corners():
    image_path = "imagini_auxiliare/01.jpg"
    image_start = cv.imread(image_path)
    image_start = cv.resize(image_start, (0, 0), fx=0.4, fy=0.4)

    image_start_hsv = cv.cvtColor(image_start, cv.COLOR_BGR2HSV)

    # Valorile experimentale sunt H: 68-255, S: 158-255, V: 139-255
    lower = np.array([68, 158, 139])
    upper = np.array([255, 255, 255])
    mask_hsv = cv.inRange(image_start_hsv, lower, upper)
    image_start_hsv = cv.bitwise_and(image_start, image_start, mask=mask_hsv)
    new_image_start = cv.cvtColor(image_start_hsv, cv.COLOR_BGR2RGB)

    new_image_start = get_closed_image(new_image_start, 64)

    new_image_start = get_open_image(new_image_start, 64)

    freq = get_freq(new_image_start)

    first_pixel = get_corner(freq, "UL")
    last_pixel = get_corner(freq, "LR")

    first_pixel = (first_pixel[0] - int(0.01 * new_image_start.shape[1]), 
                   first_pixel[1] - int(0.01 * new_image_start.shape[0]))
    
    last_pixel = (last_pixel[0] + int(0.01 * new_image_start.shape[1]),
                    last_pixel[1] + int(0.01 * new_image_start.shape[0]))

    new_image_start = add_red_corners(new_image_start, first_pixel, last_pixel, size=7)

    return first_pixel, last_pixel


def get_dominoes_image(source_bgr_image):
    source_hsv_image = cv.cvtColor(source_bgr_image, cv.COLOR_BGR2HSV)

    # Valorile experimentale sunt H: 83-255, S: 0-97, V: 222-255
    lower = np.array([83, 0, 222])
    upper = np.array([255, 97, 255])
    mask_hsv = cv.inRange(source_hsv_image, lower, upper)
    source_hsv_image = cv.bitwise_and(source_bgr_image, source_bgr_image, mask=mask_hsv)
    new_source_image = cv.cvtColor(source_hsv_image, cv.COLOR_BGR2RGB)

    return new_source_image


def get_masked_open_image(img):
    img = get_dominoes_image(img)
    img = get_open_image(img, 2)

    return img


def get_difference (current_image, previous_image, first_pixel, last_pixel):
    difference = cv.subtract(current_image, previous_image)
    difference = cv.cvtColor(difference, cv.COLOR_BGR2GRAY)
    difference = cv.threshold(difference, 200, 255, cv.THRESH_BINARY)[1]
    difference = difference[first_pixel[0]:last_pixel[0], first_pixel[1]:last_pixel[1]]
    difference = cv.resize(difference, (0, 0), fx=3, fy=3)
    difference = cv.threshold(difference, 200, 255, cv.THRESH_BINARY)[1]

    return difference


def template_matching(img, template):
    matches = []
    rotated_template = template
    for _ in range(4):
        rotated_template = cv.rotate(rotated_template, cv.ROTATE_90_CLOCKWISE)

        rotated_template = cv.resize(rotated_template, (img.shape[1], img.shape[0]))
        rotated_template = cv.threshold(rotated_template, 127, 255, cv.THRESH_BINARY)[1]

        res = cv.matchTemplate(img, rotated_template, cv.TM_CCOEFF_NORMED)

        res_value = res[0][0]

        matches.append(res_value)
    
    return max(matches)


def identify_domino_piece(source):
    template_matches_values_dict = {}
    for i in range(7):
        template = cv.imread(f"templates/{i}.PNG")

        template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

        template = cv.threshold(template, 127, 255, cv.THRESH_BINARY)[1]

        match_value = template_matching(source, template)
        template_matches_values_dict[i] = match_value
    
    max_value = max(template_matches_values_dict.values())
    if max_value > 0.07:
        return max(template_matches_values_dict.items(), key=lambda x: x[1])
    else:
        return 0, 0
    

def get_score_from_position(position):
    position = (position[0], ord(position[1]) - 64)
    if position in coordonates_to_scores:
        return coordonates_to_scores[position]
    else:
        return 0
    

def get_score_impact_of_domino_piece(first_half_position, second_half_position, first_half_domino_piece, second_half_domino_piece, current_player, player_1_points, player_2_points):
    player_1_gain = 0
    player_2_gain = 0

    score = get_score_from_position(first_half_position) + get_score_from_position(second_half_position)

    if current_player == 1:
        player_1_gain += score
        if first_half_domino_piece == second_half_domino_piece:
            player_1_gain += score
    else:
        player_2_gain += score
        if first_half_domino_piece == second_half_domino_piece:
            player_2_gain += score

    player_1_dice_number = scores_to_dice_number[player_1_points]
    player_2_dice_number = scores_to_dice_number[player_2_points]

    if player_1_dice_number == first_half_domino_piece or player_1_dice_number == second_half_domino_piece:
        player_1_gain += 3
    if player_2_dice_number == first_half_domino_piece or player_2_dice_number == second_half_domino_piece:
        player_2_gain += 3

    return player_1_gain, player_2_gain


def complete_with_white(img_grayscale, desired_size):
    height, width = img_grayscale.shape[:2]

    if height < desired_size:
        top = (desired_size - height) // 2
        bottom = desired_size - height - top
    else:
        top = bottom = 0

    if width < desired_size:
        left = (desired_size - width) // 2
        right = desired_size - width - left
    else:
        left = right = 0

    img_grayscale = cv.copyMakeBorder(img_grayscale, top, bottom, left, right, cv.BORDER_CONSTANT, value=255)

    return img_grayscale


def split_image_into_patches(image):
    height, width = image.shape
    average_size = (height + width) // 2

    image = cv.resize(image, (average_size, average_size))

    average_patch_size = average_size // 15

    size_difference_height = height / 15 - average_patch_size
    size_difference_width = width / 15 - average_patch_size

    patches = []

    patches = []
    for i_iter, i in enumerate(range(0, average_size - average_patch_size + 1, average_patch_size)):
        line_deviation = int(size_difference_height * i_iter)
        for j_iter, j in enumerate(range(0, average_size - average_patch_size + 1, average_patch_size)):
            column_deviation = int(size_difference_width * j_iter)

            line_start_index = int((i + line_deviation) * 1)
            line_end_index = int((i + average_patch_size + line_deviation) * 1)
            column_start_index = int((j + column_deviation) * 1)
            column_end_index = int((j + average_patch_size + column_deviation) * 1)

            patch = image[line_start_index:line_end_index,column_start_index:column_end_index]
            count = cv.countNonZero(patch)
            patches.append((patch, i // 126 + 1, j // 126 + 1, count, line_start_index, line_end_index, column_start_index, column_end_index))
    return patches

def get_two_highest_values(patches):
    first_highest_value = 0
    second_highest_value = 0
    first_highest_patch = None
    second_highest_patch = None
    for patch in patches:
        if patch[3] > first_highest_value:
            second_highest_value = first_highest_value
            second_highest_patch = first_highest_patch
            first_highest_value = patch[3]
            first_highest_patch = patch
        elif patch[3] > second_highest_value:
            second_highest_value = patch[3]
            second_highest_patch = patch

    return first_highest_patch, second_highest_patch


def get_average_black_border_size_enchanced(img):
    black_pixels_percentage_threshold = 0.7
    black_border_left, black_border_right, black_border_top, black_border_bottom = 0, 0, 0, 0

    for j in range(int(img.shape[1] // 3)):
        total_pixels = img.shape[0]
        black_pixels = total_pixels - cv.countNonZero(img[:, j])
        if black_pixels / total_pixels > black_pixels_percentage_threshold:
            black_border_left = j

    for j in range(img.shape[1] - 1, img.shape[1] - int(img.shape[1] // 3) - 1, -1):
        total_pixels = img.shape[0]
        black_pixels = total_pixels - cv.countNonZero(img[:, j])
        if black_pixels / total_pixels > black_pixels_percentage_threshold:
            black_border_right = img.shape[1] - j

    for i in range(int(img.shape[0] // 3)):
        total_pixels = img.shape[1]
        black_pixels = total_pixels - cv.countNonZero(img[i, :])
        if black_pixels / total_pixels > black_pixels_percentage_threshold:
            black_border_top = i

    for i in range(img.shape[0] - 1, img.shape[0] - int(img.shape[0] // 3) -1, -1):
        total_pixels = img.shape[1]
        black_pixels = total_pixels - cv.countNonZero(img[i, :])
        if black_pixels / total_pixels > black_pixels_percentage_threshold:
            black_border_bottom = img.shape[0] - i

    return black_border_left, black_border_right, black_border_top, black_border_bottom


def expand_image(img, expand_percentage, fill_value=255):
    height, width = img.shape[:2]

    new_height = int(height * (1 + expand_percentage))
    new_width = int(width * (1 + expand_percentage))

    #print(new_height, new_width)

    height_difference = new_height - height
    width_difference = new_width - width

    new_img = np.full((new_height, new_width), fill_value, dtype=np.uint8)

    new_img[height_difference // 2:height_difference // 2 + height, width_difference // 2:width_difference // 2 + width] = img

    return new_img


def middle_point(image):
    total_x = 0
    total_y = 0
    count = 0

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x] == 0:
                total_x += x
                total_y += y
                count += 1

    if count == 0:
        return None

    average_x = total_x / count
    average_y = total_y / count

    return average_x, average_y


def center_image(image, center, desired_center):
    if center is not None:
        horizontal_translation = int(desired_center[0] - center[0])
        vertical_translation = int(desired_center[1] - center[1])

        image = np.roll(image, horizontal_translation, axis=1)
        image = np.roll(image, vertical_translation, axis=0)
    
    return image


def identify_black_lines(image, threshold=50, line_length=100, line_gap=10):
    if len(image.shape) > 2:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image

    inverted = cv.bitwise_not(gray)

    edges = cv.Canny(inverted, 50, 150, apertureSize=3)

    lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold, minLineLength=line_length, maxLineGap=line_gap)

    line_image = np.zeros_like(image)

    vertical_lines_x_coordinates = []
    horizontal_lines_y_coordinates = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if x1 == x2:
                vertical_lines_x_coordinates.append(x1)
            elif y1 == y2:
                horizontal_lines_y_coordinates.append(y1)

    left_border, right_border, top_border, bottom_border = 0, 0, 0, 0

    left_values = [x for x in vertical_lines_x_coordinates if x < image.shape[1] // 2]
    if len(left_values) != 0:
        left_border = max(left_values)

    right_values = [x for x in vertical_lines_x_coordinates if x > image.shape[1] // 2]
    if len(right_values) != 0:
        right_border = image.shape[1] - min(right_values)

    top_values = [y for y in horizontal_lines_y_coordinates if y < image.shape[0] // 2]
    if len(top_values) != 0:
        top_border = max(top_values)

    bottom_values = [y for y in horizontal_lines_y_coordinates if y > image.shape[0] // 2]
    if len(bottom_values) != 0:
        bottom_border = image.shape[0] - min(bottom_values)

    return line_image, left_border, right_border, top_border, bottom_border


def apply_translation(img, source_img, left_border, right_border, top_border, bottom_border, img_line_start_index, img_line_end_index, img_column_start_index, img_column_end_index, fill_value=255):
    horizontal_translation, vertical_translation = left_border - right_border, top_border - bottom_border

    if img_line_start_index + vertical_translation >= 0 and img_line_start_index + vertical_translation < source_img.shape[0]:
        img_line_start_index += vertical_translation

    if img_line_end_index + vertical_translation >= 0 and img_line_end_index + vertical_translation < source_img.shape[0]:
        img_line_end_index += vertical_translation

    if img_column_start_index + horizontal_translation >= 0 and img_column_start_index + horizontal_translation < source_img.shape[1]:
        img_column_start_index += horizontal_translation

    if img_column_end_index + horizontal_translation >= 0 and img_column_end_index + horizontal_translation < source_img.shape[1]:
        img_column_end_index += horizontal_translation

    img = source_img[img_line_start_index:img_line_end_index, img_column_start_index:img_column_end_index]

    return img, img_line_start_index, img_line_end_index, img_column_start_index, img_column_end_index


def apply_canny_hough_lines_translation(img, source_image, img_line_start_index, img_line_end_index, img_column_start_index, img_column_end_index):
    _, left_border, right_border, top_border, bottom_border = identify_black_lines(img, line_length=int(img.shape[0] * 0.3), line_gap=int(img.shape[0] * 0.2))

    img, img_line_start_index, img_line_end_index, img_column_start_index, img_column_end_index = apply_translation(img, source_image, left_border, right_border, top_border, bottom_border, img_line_start_index, img_line_end_index, img_column_start_index, img_column_end_index)

    return img, img_line_start_index, img_line_end_index, img_column_start_index, img_column_end_index


def apply_heuristic_black_borders_translation(img, source_image, img_line_start_index, img_line_end_index, img_column_start_index, img_column_end_index):
    left_border, right_border, top_border, bottom_border = get_average_black_border_size_enchanced(img)
    
    img, img_line_start_index, img_line_end_index, img_column_start_index, img_column_end_index = apply_translation(img, source_image, left_border, right_border, top_border, bottom_border, img_line_start_index, img_line_end_index, img_column_start_index, img_column_end_index)

    return img, img_line_start_index, img_line_end_index, img_column_start_index, img_column_end_index


def get_translated_patch(patch, patch_line_start_index, patch_line_end_index, patch_column_start_index, patch_column_end_index, source_image, instructions):
    for instruction in instructions:
        if instruction == "CH":
            patch, patch_line_start_index, patch_line_end_index, patch_column_start_index, patch_column_end_index = apply_canny_hough_lines_translation(patch, source_image, patch_line_start_index, patch_line_end_index, patch_column_start_index, patch_column_end_index)
        elif instruction == "HB":
            patch, patch_line_start_index, patch_line_end_index, patch_column_start_index, patch_column_end_index = apply_heuristic_black_borders_translation(patch, source_image, patch_line_start_index, patch_line_end_index, patch_column_start_index, patch_column_end_index)
        else:
            continue

    return patch


def cut_border(img, percentage):
    height, width = img.shape[:2]

    new_height = int(height * (1 - percentage))
    new_width = int(width * (1 - percentage))

    height_difference = height - new_height
    width_difference = width - new_width

    new_img = img[height_difference // 2:height_difference // 2 + new_height, width_difference // 2:width_difference // 2 + new_width]

    return new_img


def apply_patch_filling(patch, desired_size, patch_line_start_index, patch_line_end_index, patch_column_start_index, patch_column_end_index, source_image, instructions):
    patch = get_translated_patch(patch, patch_line_start_index, patch_line_end_index, patch_column_start_index, patch_column_end_index, source_image, instructions)

    patch = expand_image(patch, 0.2)

    kernel1 = np.ones((10, 10), np.uint8)
    patch = cv.dilate(patch, kernel1, iterations=1)

    patch = get_closed_image(patch, 4)

    patch = cv.medianBlur(patch, 5)

    patch = complete_with_white(patch, desired_size)

    patch = get_closed_image(patch, 12)

    patch[:, int(patch.shape[1] * 0.95):] = 255
    patch[:, :int(patch.shape[1] * 0.05)] = 255
    patch[:int(patch.shape[0] * 0.05), :] = 255
    patch[int(patch.shape[0] * 0.95):, :] = 255

    center_of_points = middle_point(patch)
    desired_center = (patch.shape[1] // 2, patch.shape[0] // 2)

    patch = center_image(patch, center_of_points, desired_center)

    patch = cut_border(patch, 0.2)

    return patch


def get_vote_results(votes):
    max_count = max([len(votes[key]) for key in votes.keys()])

    if max_count == 0:
        return None
    
    max_keys = [key for key in votes.keys() if len(votes[key]) == max_count]

    if len(max_keys) == 1:
        return max_keys[0]
    
    else:
        max_sum = 0
        max_key = None
        for key in max_keys:
            current_sum = sum(votes[key])
            if current_sum > max_sum:
                max_sum = current_sum
                max_key = key

        return max_key


with open('paths.txt', 'r') as f:
    test_dir_path = f.readline().strip()
    results_dir_path = f.readline().strip()

if not os.path.exists(results_dir_path):
    os.makedirs(results_dir_path)
else:
    for file in os.listdir(results_dir_path):
        os.remove(f"{results_dir_path}/{file}")

file_names = os.listdir(test_dir_path)

number_of_matches = 0
for file_name in file_names:
    if int(file_name[0]) > number_of_matches:
        number_of_matches = int(file_name[0])

first_pixel_global, last_pixel_global = get_table_corners()

for i in range(1, number_of_matches + 1):
    player1_score, player2_score = 0, 0
    match_files = [file for file in file_names if file[0] == str(i)]
    images_files = [file for file in match_files if file.endswith(".jpg")]

    moves_file = [file for file in match_files if "mutari" in file][0]

    moves_file = open(f"{test_dir_path}/{moves_file}", "r")

    moves = moves_file.readlines()

    moves = [move for move in moves if move != "\n"]

    moves = [int(move[-2]) for move in moves]

    moves_images = list(zip(moves, images_files))

    previous_image, current_image = None, None

    previous_image = cv.imread(f"imagini_auxiliare/01.jpg")
    previous_image = cv.resize(previous_image, (0, 0), fx=0.4, fy=0.4)

    previous_image = get_masked_open_image(previous_image)

    for current_player, image_file in moves_images:
        result = ""
        
        current_image = cv.imread(f"{test_dir_path}/{image_file}")
        initial_current_image = cv.resize(current_image, (0, 0), fx=0.4, fy=0.4)

        current_image = get_masked_open_image(initial_current_image)

        difference = get_difference(current_image, previous_image, first_pixel_global, last_pixel_global)

        new_current_image = get_dominoes_image(initial_current_image)
        new_current_image = new_current_image[int(first_pixel_global[0]):int(last_pixel_global[0]), int(first_pixel_global[1]):int(last_pixel_global[1])]
        new_current_image = cv.resize(new_current_image, (0, 0), fx=3, fy=3)
        new_current_image = cv.cvtColor(new_current_image, cv.COLOR_BGR2GRAY)
        new_current_image = cv.threshold(new_current_image, 200, 255, cv.THRESH_BINARY)[1]

        desired_size = (difference.shape[0] // 15 + difference.shape[1] // 15) // 2

        difference = cv.medianBlur(difference, 7)

        patches = split_image_into_patches(difference)

        first_highest_patch, second_highest_patch = get_two_highest_values(patches)

        first_half_position = tuple([first_highest_patch[1], chr(first_highest_patch[2] + 64)])
        second_half_position = tuple([second_highest_patch[1], chr(second_highest_patch[2] + 64)])

        first_half_image = new_current_image[first_highest_patch[4]:first_highest_patch[5], first_highest_patch[6]:first_highest_patch[7]]
        second_half_image = new_current_image[second_highest_patch[4]:second_highest_patch[5], second_highest_patch[6]:second_highest_patch[7]]

        votes_first_half = {}
        votes_second_half = {}
        algorithms = [["HB", "CH"], ["CH", "HB"], ["HB"], ["CH"]]

        for algorithm in algorithms:
            instructions = algorithm
            first_half, second_half = first_half_image, second_half_image

            first_half = apply_patch_filling(first_half, desired_size, first_highest_patch[4], first_highest_patch[5], first_highest_patch[6], first_highest_patch[7], new_current_image, instructions)
            second_half = apply_patch_filling(second_half, desired_size, second_highest_patch[4], second_highest_patch[5], second_highest_patch[6], second_highest_patch[7], new_current_image, instructions)

            first_half_domino_piece, confidence_rate_1 = identify_domino_piece(first_half)
            second_half_domino_piece, confidence_rate_2 = identify_domino_piece(second_half)

            if first_half_domino_piece not in votes_first_half.keys():
                votes_first_half[first_half_domino_piece] = [confidence_rate_1]
            else:
                votes_first_half[first_half_domino_piece].append(confidence_rate_1)

            if second_half_domino_piece not in votes_second_half.keys():
                votes_second_half[second_half_domino_piece] = [confidence_rate_2]
            else:
                votes_second_half[second_half_domino_piece].append(confidence_rate_2)

        first_half_domino_piece = get_vote_results(votes_first_half)
        second_half_domino_piece = get_vote_results(votes_second_half)

        result += f"{first_half_position[0]}{first_half_position[1]} {first_half_domino_piece}\n"
        result += f"{second_half_position[0]}{second_half_position[1]} {second_half_domino_piece}\n"

        player_1_gain, player_2_gain = get_score_impact_of_domino_piece(first_half_position, second_half_position, first_half_domino_piece, second_half_domino_piece, current_player, player1_score, player2_score)

        player1_score += player_1_gain
        player2_score += player_2_gain

        if current_player == 1:
            result += f"{player_1_gain}\n"
        else:
            result += f"{player_2_gain}\n"

        result_file = open(f"{results_dir_path}/{image_file[:-4]}.txt", "w")
        result_file.write(result)
        result_file.close()

        previous_image = current_image