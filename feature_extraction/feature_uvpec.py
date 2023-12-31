# copying feature extraction function from UVP6 classifier
# please refer to https://github.com/ecotaxa/uvpec

from skimage import io, measure
from numpy import argmax, histogram
from math import sqrt, atan2
from collections import OrderedDict


def feature_uvpec(img, img_filename, threshold=20):
    """
    -        -
    Parameters
    ----------
    img : ndarray
        image array

    img_filename : str
        Name of the image file containing the object for features extraction.

    threshold : uint8 (0 <-> 255)
        Threshold value used to split image pixels into foreground (> threshold)
        and background (<= threshold) pixels.

    Returns
    -------
    features : OrderedDict
        Ordered Dictionary containing the features extracted from biggest
        connected region found in image.
        An empty OrderedDict is returned if no regions found.
    """


    # apply thresholding
    thresh_img = img > threshold
    if thresh_img.sum() < 1:  # there are no pixels above the threshold, return empty dict
        print("get_uvp6_features function : No objects found in",
              img_filename, "with threshold", threshold)
        return OrderedDict()

    # segmentation into connected regions
    label_img = measure.label(thresh_img)

    # get region properties for connected regions found
    props = measure.regionprops(label_img, img)

    # get index of the region presenting the biggest area in square pixels
    Areas = list()
    for region in props:
        Areas.append(region.area)
    max_area_idx = argmax(Areas)
    region = props[max_area_idx]

    # get gray values histogram for this region, and clear the 0 bin (background pixels)
    hist = histogram(region.intensity_image, bins=256, range=(0, 256))[0]
    hist[0] = 0

    # calculate histogram related features
    # (or just recover those directly available from regionprops)
    mean = region.mean_intensity
    vmin = region.min_intensity
    vmax = region.max_intensity
    intden = region.weighted_moments_central[0][0]  # this is the sum of all pixel values
    mode = argmax(hist)
    vrange = vmax - vmin
    meanpos = (mean - vmin) / vrange

    # get quartiles, and accumulate squared pixels values for stddev calculation
    nb_pixels = region.area
    first_quartile = 0.25 * nb_pixels
    second_quartile = 0.5 * nb_pixels
    third_quartile = 0.75 * nb_pixels

    square_gray_acc = 0;
    pix_acc = 0
    median = -1;
    histcum1 = -1;
    histcum3 = -1

    for gray_level, count in enumerate(hist):
        if count != 0:
            square_gray_acc += count * gray_level * gray_level
            pix_acc += count
            if (histcum1 == -1) and (pix_acc > first_quartile): histcum1 = gray_level
            if (median == -1) and (pix_acc > second_quartile): median = gray_level
            if (histcum3 == -1) and (pix_acc > third_quartile): histcum3 = gray_level

    stddev = sqrt((square_gray_acc / nb_pixels) - (mean * mean))
    cv = 100 * (stddev / mean)
    sr = 100 * (stddev / vrange)

    angle = 0.5 * atan2(2 * region.moments_central[1][1],
                        (region.moments_central[0][2] - region.moments_central[2][0]))

    # build an output ordered dict with the features vector
    # ATTENTION : feature insertion order is VERY important,
    # as it has to match exactly the feature order used on UVP6
    features = OrderedDict()
    features["area"] = nb_pixels
    features["width"] = region.bbox[3] - region.bbox[1]
    features["height"] = region.bbox[2] - region.bbox[0]
    features["mean"] = mean
    features["stddev"] = stddev
    features["mode"] = mode
    features["min"] = vmin
    features["max"] = vmax
    features["x"] = region.local_centroid[1]
    features["y"] = region.local_centroid[0]
    features["xm"] = region.weighted_local_centroid[1]
    features["ym"] = region.weighted_local_centroid[0]
    features["major"] = region.major_axis_length
    features["minor"] = region.minor_axis_length
    features["angle"] = angle
    features["eccentricity"] = region.eccentricity
    features["intden"] = intden
    features["median"] = median
    features["histcum1"] = histcum1
    features["histcum3"] = histcum3
    features["esd"] = region.equivalent_diameter
    features["range"] = vrange
    features["meanpos"] = meanpos
    features["cv"] = cv
    features["sr"] = sr
    features["bbox_area"] = region.bbox_area
    features["extent"] = region.extent

    features["central_moment-2-0"] = region.moments_central[0][2]
    features["central_moment-1-1"] = region.moments_central[1][1]
    features["central_moment-0-2"] = region.moments_central[2][0]
    features["central_moment-3-0"] = region.moments_central[0][3]
    features["central_moment-2-1"] = region.moments_central[1][2]
    features["central_moment-1-2"] = region.moments_central[2][1]
    features["central_moment-0-3"] = region.moments_central[3][0]

    """
    Current SciKit Hu Moments implementation is apparently wrong !
    (bad coordinate system convention rc <-> xy)
    It only has an impact on the sign of seventh Hu moment (mirroring)
    This is why we're inverting the sign here for hu_moment-7
    """
    features["hu_moment-1"] = region.moments_hu[0]
    features["hu_moment-2"] = region.moments_hu[1]
    features["hu_moment-3"] = region.moments_hu[2]
    features["hu_moment-4"] = region.moments_hu[3]
    features["hu_moment-5"] = region.moments_hu[4]
    features["hu_moment-6"] = region.moments_hu[5]
    features["hu_moment-7"] = - region.moments_hu[6]  # see comment above

    features["gray_central_moment-2-0"] = region.weighted_moments_central[0][2]
    features["gray_central_moment-1-1"] = region.weighted_moments_central[1][1]
    features["gray_central_moment-0-2"] = region.weighted_moments_central[2][0]
    features["gray_central_moment-3-0"] = region.weighted_moments_central[0][3]
    features["gray_central_moment-2-1"] = region.weighted_moments_central[1][2]
    features["gray_central_moment-1-2"] = region.weighted_moments_central[2][1]
    features["gray_central_moment-0-3"] = region.weighted_moments_central[3][0]

    features["gray_hu_moment-1"] = region.weighted_moments_hu[0]
    features["gray_hu_moment-2"] = region.weighted_moments_hu[1]
    features["gray_hu_moment-3"] = region.weighted_moments_hu[2]
    features["gray_hu_moment-4"] = region.weighted_moments_hu[3]
    features["gray_hu_moment-5"] = region.weighted_moments_hu[4]
    features["gray_hu_moment-6"] = region.weighted_moments_hu[5]
    features["gray_hu_moment-7"] = - region.weighted_moments_hu[6]  # see comment above

    return features



