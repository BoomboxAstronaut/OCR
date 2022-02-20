import btk
import cv2 as cv
import numpy as np
import tensorflow as tf
from PIL import ImageGrab, PngImagePlugin

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

def sp_predict(slices, sens):
    """Segmentation model"""
    preds = np.array([x[0] for x in tseg.predict(slices)])
    preds[preds < sens] = 0
    return preds

def extract_rows(imset, font_scale: float = 1):
    """Extract rows of text from areas containing text"""
    img = imset
    blur_coef_1, blur_coef_2 = btk.odder(3 * font_scale), btk.odder(1 * font_scale)
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
    bounds = [x for x in bounds if btk.find_area(x) > 512 * font_scale and x[1] - x[0] > 6 * font_scale]
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
    img = btk.force_dim(img, dims[0], 1)
    hist = np.histogram(img.flatten(), 255, (0, 255))
    levels_low = int(hist[0].argmax())
    levels_high = int(max([x for i, x in enumerate(hist[1]) if hist[0][i - 1] >= 7]))
    img[img < levels_low] = 0
    original_img = img.copy()
    buffer = int((levels_high - levels_low) * 0.333)
    img[img < levels_low + buffer] = 0
    img[img > 0] = 255
    slices = btk.img_slicer(img, dims, 1, 1)
    idx = btk.gen_index(img.shape, dims, 1, 1)
    predictions = sp_predict(slices, sens1)
    predictions = np.array(btk.smooth_avg(btk.smooth_avg(predictions)))
    predictions[predictions < sens2] = 0
    crits = btk.criticals(predictions, idx)
    crits = [int(btk.halfpoint(x[0][0], x[0][1])) for x in crits if x[1] == 'max']
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
    img = cv.GaussianBlur(img, (btk.odder(5 * font_scale), 1), 150)
    img[img > 0] = 255
    conts = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    bounds = []
    for x in conts:
        xset = [y[0][0] for y in x]
        yset = [y[0][1] for y in x]
        bounds.append((min(yset), max(yset), min(xset), max(xset)))
    bounds = [x for x in bounds if btk.find_area(x) > 64 * font_scale and x[1] - x[0] > 6 * font_scale]
    bounds = expand_coordinates(bounds, oimg[1].shape, (0.12, 0.06))
    bounds = sorted(bounds, key=lambda x: x[2])
    bounds = {(x[0] + oimg[0][0], x[1] + oimg[0][0], x[2] + oimg[0][2], x[3] + oimg[0][2]): oimg[1][x[0]:x[1], x[2]:x[3]] for x in bounds}
    return bounds

def readimg(img, spacing_sens_1: float = 0.5, spacing_sens_2: float = 0.9, extract_sens_1: int = 1, extract_sens_2: int = 1):
    """Primary wrapper function"""
    rows = extract_rows(btk.grey_np(img), extract_sens_1)
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

if __name__ == '__main__':
    if isinstance(ImageGrab.grabclipboard(), PngImagePlugin.PngImageFile):
        chars = list('abcdefghijklmnopqrstuvwxyzABDEFGHIJKLMNQRTUY0123456789 ():;.,"\'!@#$%&?+=-')
        tdet = tf.keras.models.load_model('models\\tdet-3')
        tseg = tf.keras.models.load_model('models\\tseg-0')
        tcls = tf.keras.models.load_model('models\\tcls-3')
        print('Processing...')
        readimg(btk.grey_np(ImageGrab.grabclipboard()), 0.5, 0.9, 1, 1)
    else:
        print('Load image into clipboard with a screenshot to use')
else:
    pass
