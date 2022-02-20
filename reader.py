"""
Converts text in images into strings.

Reads from clipboard
Models must be in the same directory
"""
import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from PIL import Image, ImageGrab, PngImagePlugin

def expand_coordinates(bounds: tuple[int, ...], idims: tuple[int, int], stretch: tuple[float, float] = (0.085, 0.085)) -> tuple[int, ...]:
    """Expands coordinate inputs slightly to account for imprecise text detection"""
    for i, x in enumerate(bounds):
        ygap = (x[1] - x[0]) * stretch[0]
        xgap = (x[3] - x[2]) * stretch[1]
        bounds[i] = [round(x[0] - (ygap * 1.5)), round(x[1] + (ygap * 0.5)), round(x[2] - xgap), round(x[3] + xgap)]
    for i, x in enumerate(bounds):
        if x[0] < 0:
            bounds[i][0] = 0
        if x[1] > idims[0]:
            bounds[i][1] = idims[0]
        if x[2] < 0:
            bounds[i][2] = 0
        if x[3] > idims[1]:
            bounds[i][3] = idims[1]
    return bounds

def find_area(coords):
    """Find area of a box when given the box corners in a tuple"""
    return (coords[1] - coords[0]) * (coords[3] - coords[2])

def sp_predict(slices, sens):
    """Segmentation model"""
    preds = np.array([x[0] for x in tseg.predict(slices)])
    preds[preds < sens] = 0
    return preds

def extract_rows(imset, font_scale: float = 1):
    """Extract rows of text from areas containing text"""
    img = imset
    blur_coef_1, blur_coef_2 = odder(3 * font_scale), odder(1 * font_scale)
    img = cv.Canny(img, 50, 200, apertureSize=3)
    img = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=1)
    img = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=1)
    img = np.uint8(np.absolute(img))
    img = cv.GaussianBlur(img, (blur_coef_1, blur_coef_1), 35)
    img = cv.GaussianBlur(img, (blur_coef_1, blur_coef_2), 35)
    img = cv.GaussianBlur(img, (blur_coef_1, blur_coef_2), 35)
    img[img > 0] = 255
    conts = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    bounds = []
    for x in conts:
        xset = [y[0][0] for y in x]
        yset = [y[0][1] for y in x]
        bounds.append((min(yset), max(yset), min(xset), max(xset)))
    #Filter
    bounds = [x for x in bounds if find_area(x) > 512 * font_scale and x[1] - x[0] > 6 * font_scale]
    bounds = expand_coordinates(bounds, imset.shape, (0.08, 0.02))
    bounds = {(x[0], x[1], x[2], x[3]): imset[x[0]:x[1], x[2]:x[3]] for x in bounds}
    for x in bounds.copy():
        #Filters
        img = bounds.get(x)
        if len([x for x in np.histogram(img.flatten(), 255, (0, 255))[0] if x > 3]) >= 176:
            bounds.pop(x)
        else:
            img = cv.Canny(img, 50, 200, apertureSize=3)
            img = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=1)
            img = np.uint8(np.absolute(img))
            if np.sum(img) / img.size < 32:
                bounds.pop(x)
    return bounds

def chopper(img_1, critical_points: list):
    """Returns images of sliced """
    critical_points.sort()
    sliced_letters = {}
    critical_points.insert(0, 0)
    if critical_points[-1] - len(critical_points) >= 32:
        critical_points.append(critical_points[-1] + 32)
    else:
        critical_points.append(len(img_1[0]) - 1)
    for i, x in enumerate(critical_points[:-1].copy()):
        if 64 <= critical_points[i + 1] - x < 128:
            critical_points.append(round(((critical_points[i + 1] - x) / 2) + x))
        if 128 <= critical_points[i + 1] - x < 192:
            critical_points.append(round(((critical_points[i + 1] - x) / 3) + x))
            critical_points.append(round((((critical_points[i + 1] - x) / 3) * 2) + x))
    critical_points = sorted(critical_points)
    for i, x in enumerate(critical_points[:-1]):
        sliced_letters.update({i: img_1[:, x:critical_points[i + 1]]})
    for x in sliced_letters.copy().items():
        if x[1].shape[1] < 3 or x[1].shape[1] >= 65:
            sliced_letters.pop(x[0])
    return sliced_letters.values()

def split_letters(img, dims, sens1, sens2):
    """Seperate letters inside words"""
    img = force_dim(img, dims[0], 1)
    hist = np.histogram(img.flatten(), 255, (0, 255))
    levels_low = int(hist[0].argmax())
    levels_high = int(max([x for i, x in enumerate(hist[1]) if hist[0][i - 1] >= 7]))
    img[img < levels_low] = 0
    original_img = img.copy()
    buffer = int((levels_high - levels_low) * 0.333)
    img[img < levels_low + buffer] = 0
    img[img > 0] = 255
    slices = img_slicer(img, dims, 1, 1)
    idx = gen_index(img.shape, dims, 1, 1)
    predictions = sp_predict(slices, sens1)
    predictions = np.array(smooth_avg(smooth_avg(predictions)))
    predictions[predictions < sens2] = 0
    crits = criticals(predictions, idx)
    crits = [int(halfpoint(x[0][0], x[0][1])) for x in crits if x[1] == 'max']
    original_img[original_img < levels_low + int((levels_high - levels_low) * 0.45)] = 0
    original_img[original_img > 0] = 255
    if len(crits) >= 1:
        letters = chopper(original_img, crits)
    elif img.shape[1] <= 64:
        letters = [img]
    else:
        return
    background = np.zeros((64, 64), dtype='uint8')
    finished = []
    for x in letters:
        xstart = round((64 - len(x[0])) / 2)
        ystart = round((64 - len(x)) / 2)
        formatted_letter = background.copy()
        formatted_letter[ystart:ystart + len(x), xstart:xstart + len(x[0])] = x
        finished.append(formatted_letter)
    return np.array(finished)

def extract_words(oimg, font_scale: int = 1):
    """Extract words from sentences"""
    img = cv.Canny(oimg[1], 50, 200, apertureSize=3)
    img = cv.GaussianBlur(img, (odder(5 * font_scale), 1), 150)
    img[img > 0] = 255
    conts = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    bounds = []
    for x in conts:
        xset = [y[0][0] for y in x]
        yset = [y[0][1] for y in x]
        bounds.append((min(yset), max(yset), min(xset), max(xset)))
    bounds = [x for x in bounds if find_area(x) > 64 * font_scale and x[1] - x[0] > 6 * font_scale]
    bounds = expand_coordinates(bounds, oimg[1].shape, (0.12, 0.06))
    bounds = sorted(bounds, key=lambda x: x[2])
    bounds = {(x[0] + oimg[0][0], x[1] + oimg[0][0], x[2] + oimg[0][2], x[3] + oimg[0][2]): oimg[1][x[0]:x[1], x[2]:x[3]] for x in bounds}
    return bounds

def readimg(img, spacing_sens_1: float = 0.5, spacing_sens_2: float = 0.9, extract_sens_1: int = 1, extract_sens_2: int = 1):
    """Primary wrapper function"""
    rows = extract_rows(grey_np(img), extract_sens_1)
    all_words = []
    for row in rows.items():
        words =  extract_words(row, extract_sens_2)
        sentence = []
        for word in words.items():
            letters = split_letters(word[1], (64, 12), spacing_sens_1, spacing_sens_2)
            if letters is not None:
                predictions = tcls.predict(letters)
                word_string = []
                for prediction in predictions:
                    if sum(sorted(prediction)[-3:]) > 0.9:
                        word_string.append(chars[prediction.argmax()])
                sentence.append([''.join(word_string), list(word[0])])
        all_words.extend(sentence)
    current_position = all_words[0][1][0]
    for i, x in enumerate(all_words):
        if abs(current_position - x[1][0]) < 5:
            all_words[i][1][0] = current_position
        else:
            current_position = x[1][0]
    for x in set(y[1][0] for y in all_words):
        print(' '.join([z[0].strip() for z in sorted(all_words, key=lambda x: x[1][2]) if z[1][0] == x]).lower())

def odder(num: int) -> int:
    """Forces a number to be odd"""
    if num % 2 == 0:
        num += 1
    return int(num)

def grey_np(img: np.ndarray) -> np.ndarray:
    """
    Return a greyscale version of the input image

    Args:
        img (Image): Color Image

    Returns:
        np.ndarray: Greyscale Image
    """
    img = np.array(img)
    if len(img.shape) == 2:
        return img
    if img.shape[2] == 4:
        greyscaler = [0.21, 0.72, 0.07, 0]
    elif img.shape[2] == 3:
        greyscaler = [0.21, 0.72, 0.07]
    else:
        print("Invalid Image Colors")
        return None
    return np.dot(np.array(img), greyscaler).astype('uint8')

def smooth_avg(data: list[int]) -> list[int]:
    """
    Generate a smoothed version of a data set where each point is replaced by the average of itself and immeadiately adjacent points

    Args:
        data (list[int]): A list of continuous data points

    Returns:
        list[int]: A smoother list of continuous data points
    """
    smoothed = []
    for i, x in enumerate(data):
        if i == 0:
            smoothed.append((x + data[i + 1]) / 2)
        elif i == len(data) - 1:
            smoothed.append((x + data[i - 1]) / 2)
        else:
            smoothed.append((data[i - 1] + x + data[i + 1]) / 3)
    return smoothed

def criticals(data: list, idx: bool = False) -> list:
    """
    Create a list of critical points of a continuous data set
    Critical Points: Maxima, Minima, Gradient Maxima, Gradient Minima, Gradient Roots

    Args:
        data (list): A list of continuous data points
        idx (bool, optional): A custom index. Defaults to False.

    Returns:
        list: A list of tuples that contains the index of a critical point and the critical point type
    """
    grads = np.gradient(data)
    grads2 = np.gradient(grads)
    crits = []
    if not idx:
        idx = range(len(data))
    """ else:
        idx = [round((x[1] - x[0]) / 2) + x[0] for x in idx] """
    for i, x in enumerate(idx, 1):
        if i > len(idx) - 2:
            break
        if data[i - 1] < data[i] and data[i + 1] < data[i]:
            crits.append((x, 'max'))
        if data[i - 1] > data[i] and data[i + 1] > data[i]:
            crits.append((x, 'min'))
        if grads[i] > 0 and grads[i + 1] < 0 or grads[i] < 0 and grads[i + 1] > 0:
            crits.append((x, 'dzero'))
        if grads[i - 1] < grads[i] and grads[i + 1] < grads[i]:
            crits.append((x, 'dmax'))
        if grads[i - 1] > grads[i] and grads[i + 1] > grads[i]:
            crits.append((x, 'dmin'))
        if grads2[i] > 0 and grads2[i + 1] < 0 or grads2[i] < 0 and grads2[i + 1] > 0:
            crits.append((x, 'ddzero'))
        if grads2[i - 1] < grads2[i] and grads2[i + 1] < grads2[i]:
            crits.append((x, 'ddmax'))
        if grads2[i - 1] > grads2[i] and grads2[i + 1] > grads2[i]:
            crits.append((x, 'ddmin'))
    return crits

def img_slicer(img: np.ndarray, sdims: tuple[int, int], step: int or tuple[int, int], axis: int) -> tuple[np.ndarray, list]:
    """
    Slice an image into smaller pieces

    Args:
        sdims (tuple[int, int]): Dimensions of a image slice (y axis, x axis)
        img (Image): Image
        step (intortuple[int, int]): Step distance between slices (y axis, x axis)
        axis (int): String indicating slicing method
            1 to slice using only verical cuts
            0 to slice using only horizontal cuts
            2 to slice in both orientations

    Returns:
        tuple[np.ndarray, list]: Array of image slices as array objects and a list of the boundry coordinates for each slice
    """
    slices = []
    assert axis in [0, 1, 2], 'Invalid Slicing Method'
    if axis == 1:
        shape, border = img.shape, round(sdims[1] / 2)
        for x in range(border, shape[1] - border, step):
            slices.append(img[:, x - border:x + border])
    elif axis == 0:
        shape, border = img.shape, round(sdims[0] / 2)
        for y in range(border, shape[0] - border, step):
            slices.append(img[y - border:y + border, :])
    elif axis == 2:
        assert isinstance(step, tuple), 'Step must be tuple for 2 way slicing'
        shape, yborder, xborder = img.shape, round(sdims[0] / 2), round(sdims[1] / 2)
        for y in range(yborder, shape[0] - round(yborder / 2), step[0]):
            for x in range(xborder, shape[1] - round(xborder / 2), step[1]):
                slices.append(img[y - yborder:y + yborder, x - xborder:x + xborder])
    return np.array(slices)

def gen_index(dims: tuple[int, int], sdims: tuple[int, int], step: int, axis: int) -> list[tuple[int, int]]:
    """Create custom index for image slices that correspond to pixel coordinates"""
    assert axis in [0, 1, 2], 'Invalid Axis'
    if axis == 1:
        border = round(sdims[1] / 2)
        idx = [(int(x - border), int(x + border)) for x in range(border, dims[1] - border, step)]
    elif axis == 0:
        border = round(sdims[0] / 2)
        idx = [(int(x - border), int(x + border)) for x in range(border, dims[0] - border, step)]
    elif axis == 2:
        yborder, xborder = round(sdims[0] / 2), round(sdims[1] / 2)
        idx = []
        for y in range(yborder, dims[0] - round(yborder / 2), step[0]):
            for x in range(xborder, dims[1] - round(xborder / 2), step[1]):
                idx.append((
                    int(y - round(sdims[0] / 2)),
                    int(y + round(sdims[0] / 2)),
                    int(x - round(sdims[1] / 2)),
                    int(x + round(sdims[1] / 2))
                ))
    return idx

def force_dim(img: np.ndarray, dim: int, axis: int) -> np.ndarray:
    """
    Resize an image forcing one dimension to the input dimension and scaling the other dimension by the same factor

    in and out both np arrays

    Args:
        img (np.ndarray): Input image
        dim (tuple[int, int]): Dimensions to scale image to
        axis (int): Principal scaling axis

    Returns:
        np.ndarray: Rescaled image
    """
    assert axis in [1, 0], 'Invalid Axis'
    img = Image.fromarray(img)
    if axis == 1:
        img = img.resize((round(dim / img.size[1] * img.size[0]), dim))
    elif axis == 0:
        img = img.resize((dim, round(dim / img.size[1] * img.size[0])))
    return np.array(img)

def halfpoint(num1: int or float, num2: int or float):
    """
    Gives the halfway point between input numbers

    Args:
        num1 (intorfloat): A number
        num2 (intorfloat): A number

    Returns:
        [type]: Halfway point number
    """
    if num2 > num1:
        mid = ((num2 - num1) / 2) + num1
    else:
        mid = ((num1 - num2) / 2) + num2
    return mid

if __name__ == '__main__':
    if isinstance(ImageGrab.grabclipboard(), PngImagePlugin.PngImageFile):
        print('Loading Models...')
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        chars = list('abcdefghijklmnopqrstuvwxyzABDEFGHIJKLMNQRTUY0123456789 ():;.,"\'!@#$%&?+=-')
        tdet = tf.keras.models.load_model('models\\tdet-3')
        tseg = tf.keras.models.load_model('models\\tseg-0')
        tcls = tf.keras.models.load_model('models\\tcls-3')
        print('Processing...')
        readimg(grey_np(ImageGrab.grabclipboard()), 0.5, 0.9, 1, 1)
    else:
        print('Load image into clipboard with a screenshot to use')
else:
    pass
