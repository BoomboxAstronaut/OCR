import btk
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

tdetect = tf.keras.models.load_model('models\\tdetect-3')
tseg = tf.keras.models.load_model('models\\tseg')
ccls = tf.keras.models.load_model('models\\ccls-h')
chars = list('abcdefghijklmnopqrstuvwxyzABDEFGHIJKLMNQRTUY0123456789')


def txt_detect(img: np.ndarray, sdims: tuple[int, int], sens: float, sfactor: float = True) -> tuple[int, ...]:
    """
    Use ML model to determine if text appears within an image

    Args:
        img (np.ndarray): Image to analyze
        sens (float): Sensitivity threshold for text detection 

    Returns:
        list[tuple[int, ...], np.ndarray]: Coordinates of regions that have text within the image
    """
    if sfactor:
        sfactor = img.size
    oshape = img.shape
    sc_coeff = (sfactor / img.size)**0.5
    img = btk.resize(img, (round(img.shape[0] * sc_coeff), round(img.shape[1] * sc_coeff)))
    img = btk.fit2dims(img, sdims)
    coord_scale = ((oshape[0] / img.shape[0], oshape[1] / img.shape[1]))
    slices = btk.img_slicer(img, sdims, sdims, 'm')
    coords = btk.gen_index(img.shape, sdims, sdims, 'm')
    predictions = tdetect.predict(slices)
    tcords = []
    for i, x in enumerate(predictions):
        if x > sens:
            tcords.append((coords[i], x[0]))
    for i, x in enumerate(tcords):
        tcords[i] = (np.array([x[0][0] * coord_scale[0],
                            x[0][1] * coord_scale[0],
                            x[0][2] * coord_scale[1],
                            x[0][3] * coord_scale[1]]).astype('uint16'), x[1])
    return tcords

def txt_heat_mapper(img: np.ndarray, sdims: tuple[int, int], sens: float) -> np.ndarray:
    """Generates a heatmap for text locations on an image"""
    img = btk.fit2dims(img, sdims)
    iarea = img.size
    score_card = np.zeros((img.shape[0], img.shape[1]), dtype='float16')
    if iarea < 1048576:
        cords = txt_detect(img, sdims, sens - ((1 - sens) - ((1 - sens) ** 1.666)), iarea * 4)
        for x in cords:
            score_card[x[0][0]:x[0][1], x[0][2]:x[0][3]] = score_card[x[0][0]:x[0][1], x[0][2]:x[0][3]] + (np.ones((x[0][1] - x[0][0], x[0][3] - x[0][2])) * x[1])
    if iarea < 2097152:
        cords = txt_detect(img, sdims, sens - ((1 - sens) - ((1 - sens) ** 1.333)), iarea * 2)
        for x in cords:
            score_card[x[0][0]:x[0][1], x[0][2]:x[0][3]] = score_card[x[0][0]:x[0][1], x[0][2]:x[0][3]] + (np.ones((x[0][1] - x[0][0], x[0][3] - x[0][2])) * x[1])
    cords = txt_detect(img, sdims, sens, iarea)
    for x in cords:
        score_card[x[0][0]:x[0][1], x[0][2]:x[0][3]] = score_card[x[0][0]:x[0][1], x[0][2]:x[0][3]] + (np.ones((x[0][1] - x[0][0], x[0][3] - x[0][2])) * x[1])
    if iarea > 2097152:
        cords = txt_detect(img, sdims, sens ** 0.666, iarea / 2)
        for x in cords:
            score_card[x[0][0]:x[0][1], x[0][2]:x[0][3]] = score_card[x[0][0]:x[0][1], x[0][2]:x[0][3]] + (np.ones((x[0][1] - x[0][0], x[0][3] - x[0][2])) * x[1])
    if iarea > 4194304:
        cords = txt_detect(img, sdims, sens ** 0.333, iarea / 4)
        for x in cords:
            score_card[x[0][0]:x[0][1], x[0][2]:x[0][3]] = score_card[x[0][0]:x[0][1], x[0][2]:x[0][3]] + (np.ones((x[0][1] - x[0][0], x[0][3] - x[0][2])) * x[1])
    return score_card

def expand_coordinates(bounds: tuple[int, ...], idims: tuple[int, int], stretch: tuple[float, float] = (0.085, 0.085)) -> tuple[int, ...]:
    """Expands coordinate inputs slightly to account for imprecise text detection"""
    for i, x in enumerate(bounds):
        ygap = (x[1] - x[0]) * stretch[0]
        xgap = (x[3] - x[2]) * stretch[1]
        bounds[i] = [round(x[0] - ygap), round(x[1] + ygap), round(x[2] - xgap), round(x[3] + xgap)]
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

def textract_images(img: np.ndarray, coords: tuple[int, ...]) -> list[np.ndarray]:
    """Extract image regions defineed by coordinate list input"""
    imarr = np.array(img)
    extracted = [imarr[x[0]:x[1], x[2]:x[3]] for x in coords]
    return extracted

def merge_coords(cords: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Merge touching coordinate boxes horizontally and vertically"""
    cords.sort()
    xcombined = []
    ycombined = []
    bay = cords.pop(0)
    while cords:
        if cords[0][0] == bay[0] and cords[0][2] == bay[3]:
            bay[3] = cords[0][3]
            cords.pop(0)
        else:
            xcombined.append(bay)
            bay = cords.pop(0)
    xcombined.append(bay)
    xcombined.sort()
    bay = xcombined.pop(0)
    while xcombined:
        #if xcombined[0][0] == bay[1] and bay[2] == xcombined[0][2] and xcombined[0][3] == bay[3]:
        #    bay[1] = xcombined[0][1]
        #    xcombined.pop(0)
        if xcombined[0][0] == bay[1] and bay[3] > xcombined[0][2] and xcombined[0][3] > bay[2]:
            bay[1] = xcombined[0][1]
            bay[2] = min(xcombined[0][2], bay[2])
            bay[3] = max(xcombined[0][3], bay[3])
            xcombined.pop(0)
        else:
            ycombined.append(bay)
            bay = xcombined.pop(0)
    ycombined.append(bay)
    return ycombined

def sp_predict(slices, sens):
    preds = np.array([x[0] for x in tseg.predict(slices)])
    preds[preds < sens] = 0
    return preds

def trimmer(iar):
    img = np.array(iar)
    hgram = np.histogram(img.flatten(), 255, (0, 255))
    pix_mode = hgram[0].argmax()
    if pix_mode > 127:
        img = np.invert(img)
        pix_mode = 255 - pix_mode
    img = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 29)
    img = np.invert(img)
    img = cv.medianBlur(img, 3)
    img[img < 127] = 0
    img[img > 0] = 255
    img = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=-1)
    img = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=1)
    img = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=1)
    img = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=-1)
    img = np.uint8(np.absolute(img))
    img = cv.medianBlur(img, 3)
    img = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=-1)
    img = np.uint8(np.absolute(img))
    img = cv.erode(img, np.ones((2, 2)), iterations=1)
    img = cv.GaussianBlur(img, (9, 1), 9, 1)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=2)
    img[img > 0] = 255
    img = cv.GaussianBlur(img, (7, 1), 7, 1)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
    img[img > 0] = 255
    bounds = contour_bounds(img)
    bounds = expand_coordinates(bounds, iar.shape, (0.06, 0.03))
    for x in bounds.copy():
        if x[1] - x[0] + 6 > x[3] - x[2]:
            if x[3] - x[2] < 128:
                bounds.remove(x)
    for x in bounds.copy():
        for y in bounds.copy():
            if x[0] > y[0] and x[1] < y[1] and x[2] > y[2] and x[3] < y[3]:
                try:
                    bounds.remove(x)
                except: ValueError
    return [(iar[x[0]:x[1], x[2]:x[3]], x) for x in bounds]

def get_text(img: np.ndarray, sdims: tuple[int, int], sens: float) -> list[np.ndarray]:
    """
    Identify and extract image regions containing text

    Args:
        img (np.ndarray): Image
        sdims (tuple[int, int]): Dimensions of the image slices
        sens (float): Sensitivity of the detection model

    Returns:
        list[np.ndarray]: Image regions containing text in array form
    """
    iar = btk.grey_np(img)
    imo = btk.sharpen(iar)
    mtrx = txt_heat_mapper(imo, sdims, sens)
    img = np.invert(imo)
    mtrx += txt_heat_mapper(img, sdims, sens)
    img = cv.adaptiveThreshold(imo, 254, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 19, 30)
    mtrx += txt_heat_mapper(img, sdims, ((1 - sens) * 0.5) + sens)
    mtrx = np.array(mtrx, dtype='uint8')
    hgram = np.histogram(mtrx.flatten(), 9)
    mtrx[mtrx < hgram[1][hgram[0].argmin()] - 1] = 0
    mtrx[mtrx > 0] = 255
    mtrx = cv.GaussianBlur(mtrx, (67, 31), 49)
    mtrx[mtrx > 0] = 255
    boxes = contour_bounds(mtrx)
    for x in boxes.copy():
        for y in boxes.copy():
            if x[0] > y[0] and x[1] < y[1] and x[2] > y[2] and x[3] < y[3]:
                try:
                    boxes.remove(x)
                except: ValueError
    iar = btk.fit2dims(iar, (64, 64))
    boxes = [(iar[x[0]:x[1], x[2]:x[3]], x) for x in boxes]
    txt_areas = {}
    for x in boxes:
        for y in trimmer(x[0]):
            txt_areas.update({(y[1][0] + x[1][0], y[1][1] + x[1][0], y[1][2] + x[1][2], y[1][3] + x[1][2]): y[0]})
    return txt_areas

def contour_bounds(img) -> list[tuple[np.ndarray, list]]:
    conts = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    bounds = []
    for x in conts:
        xset = [y[0][0] for y in x]
        yset = [y[0][1] for y in x]
        bounds.append((min(yset), max(yset), min(xset), max(xset)))
        bounds = [x for x in bounds if (x[1] - x[0]) * (x[3] - x[2]) > 512 and x[1] - x[0] > 6]
    return bounds

def extract_rows(imset):
    img = imset[1]
    blur_coef_1 = 7
    if img.size > 49152:
        blur_coef_1 = btk.odder(round(7 * (img.size /  32768) ** 0.5))
    hgram = np.histogram(img.flatten(), 255, (0, 255))
    pix_mode = hgram[0].argmax()
    if pix_mode > 127:
        img = np.invert(img)
        pix_mode = 255 - pix_mode
    img = cv.Canny(img, pix_mode + 28, 200, apertureSize=3)
    if img.size > 32768:
        img = cv.medianBlur(img, 3)
        img = cv.GaussianBlur(img, (5, 1), 5)
        img = cv.GaussianBlur(img, (1, 3), 3)
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, np.ones((2, 2), np.uint8))
        img[img > 0] = 255
    img[img < 127] = 0
    img[img > 0] = 255
    img = cv.GaussianBlur(img, (blur_coef_1, 1), blur_coef_1)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    img = cv.GaussianBlur(img, (3, 1), 3)
    img[img > 0] = 255
    if np.sum(img) / (img.size * 255) < 0.1 or np.sum(img) / (img.size * 255) > 0.75:
        return
    conts = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    bounds = []
    for x in conts:
        xset = [y[0][0] for y in x]
        yset = [y[0][1] for y in x]
        bounds.append((min(yset), max(yset), min(xset), max(xset)))
    bounds = expand_coordinates(bounds, imset[1].shape, (0.08, 0.03))
    bounds = [x for x in bounds if (x[1] - x[0]) * (x[3] - x[2]) > 512 and x[1] - x[0] > 6]
    return {(imset[0][0] + x[0], imset[0][0] + x[1], imset[0][2] + x[2], imset[0][2] + x[3]): imset[1][x[0]:x[1], x[2]:x[3]] for x in bounds}

def sat_check(img):
    img = cv.Canny(img, 50, 200, apertureSize=3)
    img = cv.GaussianBlur(img, (9, 1), 9)
    img = cv.GaussianBlur(img, (1, 5), 5)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=4)
    img[img > 0] = 255
    if np.sum(img) / (img.size * 255) > 0.5:
        return True

def dedupe(texts, img):
    for z in texts.copy().items():
        if z[1].size < 32768:
            if not sat_check(z[1]):
                texts.pop(z[0])
    for z1 in texts.copy().keys():
        for z2 in texts.copy().keys():
            if z1[1] > z2[0] and z1[0] < z2[1] and z1[3] > z2[2] and z1[2] < z2[3] and z2 != z1:
                overlap = (min(z1[1], z2[1]) - max(z1[0], z2[0])) * (min(z1[3], z2[3]) - max(z1[2], z2[2]))
                overlap_1 = overlap / ((z1[3] - z1[2]) * (z1[1] - z1[0]))
                overlap_2 = overlap / ((z2[3] - z2[2]) * (z2[1] - z2[0]))
                if overlap_1 > 0.5 or overlap_2 > 0.5:
                    if overlap_2 > overlap_1:
                        outer = z1
                        inner = z2
                    else:
                        outer = z2
                        inner = z1
                    olcoef = min(overlap_1, overlap_2) / max(overlap_1, overlap_2)
                    if olcoef <= 0.15:
                        try:
                            texts.pop(inner)
                        except: ValueError
                    elif 0.15 < olcoef < 0.666:
                        try:
                            texts.pop(outer)
                        except: ValueError
                    elif olcoef > 0.666:
                        try:
                            texts.pop(outer)
                            texts.pop(inner)
                        except: ValueError
                        texts.update({(min(z1[0], z2[0]), max(z1[1], z2[1]), min(z1[2], z2[2]), max(z1[3], z2[3])):
                                    img[min(z1[0], z2[0]):max(z1[1], z2[1]), min(z1[2], z2[2]):max(z1[3], z2[3])]})
    return texts

def draw_sep(img, seps):
    img = Image.fromarray(img)
    ymax = img.size[1]
    for x in seps:
        ImageDraw.Draw(img).line([x, 0, x, ymax], 0, 1)
    return img

def chopper(img, inp: list, min=1, dist=3):
    img = np.array(img)
    inp.sort()
    last = 0
    vals = []
    confirmed = []
    for x in inp:
        if x - last >= dist:
            vals.append(last)
            if len(vals) >= min:
                confirmed.append(round(sum(vals) / len(vals)))
            vals = []
            last = x
        else:
            vals.append(last)
            last = x
    confirmed.append(last)
    if confirmed[-1] - len(confirmed) >= 32:
        confirmed.append(confirmed[-1] + 32)
    else:
        confirmed.append(len(img[0]) - 1)
    cpack = []
    for i, x in enumerate(confirmed):
        if i == len(confirmed) - 1:
            break
        if confirmed[i + 1] - x >= 4:
            cpack.append(img[:, x:confirmed[i + 1]])
    for x in cpack.copy():
        if x.shape[1] / x.shape[0] > 2:
            cpack.remove(x)
    for i, x in enumerate(cpack):
        if x.shape[1] > 64 or x.shape[0] > 64:
            cpack[i] = btk.fit2dims(x, (64, 64))
    return cpack

def split_letters(img, dims, sens):
    img = btk.force_dim(img, dims[0], 1)
    slices = btk.img_slicer(btk.sharpen(img), dims, 1, 'v')
    idx = btk.gen_index(img.shape, dims, 1, 'v')
    preds = sp_predict(slices, sens)
    preds = np.array(btk.smooth_avg(btk.smooth_avg(btk.smooth_avg(btk.smooth_avg(preds)))))
    if sum(preds) / len(preds) < 0.15 and max(preds) < 0.8:
        return
    crits = btk.criticals(preds, idx)
    crits = [btk.halfpoint(x[0][0], x[0][1]) for x in crits if x[1] == 'max']
    crit_peaks = np.array([preds[int(x) - 5] for x in crits])
    peak_avg = np.average(crit_peaks)
    stdv = (np.sum([(x - peak_avg)**2 for x in crit_peaks]) / len(crit_peaks))**0.5
    crit_peaks[crit_peaks < peak_avg - (3 * stdv)] = 0
    crits2 = [int(x) for i, x in enumerate(crits) if crit_peaks[i] > sens]
    bgcolor = round(np.mean(img) / 2)
    bg = np.ones((64, 64), dtype='uint8') * bgcolor
    letters = chopper(img, crits2)
    finished = []
    for x in letters:
        if len(x) < 38 and len(x[0]) < 43:
            x = btk.force_dim(x, 54, 1)
        xstart = round((64 - len(x[0])) / 2)
        ystart = round((64 - len(x)) / 2)
        xbg = bg.copy()
        xbg[ystart:ystart + len(x), xstart:xstart + len(x[0])] = x
        #xbg = cv.adaptiveThreshold(xbg, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 9)
        #xbg = np.invert(xbg)
        xbg = cv.GaussianBlur(xbg, (5, 5), 5)
        xbg = btk.sharpen(xbg)
        finished.append(xbg)
    return finished

def search_read(img, sens1, sens2):
    img = np.asarray(img)
    for tarea in get_text(img, (64, 64), sens1).items():
        texts = extract_rows(tarea)
        if texts:
            img = btk.fit2dims(btk.grey_np(img), (64, 64))
            rows = dedupe(texts, img)
            for row in rows.items():
                lpack = split_letters(row[1], (32, 12), sens2)
                if lpack:
                    ctest = ccls.predict(np.array(lpack))
                    word = []
                    for letter in ctest:
                        word.append(chars[letter.argmax()])
                    print(''.join(word).lower())

def readf(img, sens):
    img = btk.grey_np(img)
    lpack = split_letters(img, (32, 12), sens)
    if lpack:
        ctest = ccls.predict(np.array(lpack))
        word = []
        for i, letter in enumerate(ctest):
            word.append(chars[letter.argmax()])
        word.pop(0)
        print(''.join(word).lower())

