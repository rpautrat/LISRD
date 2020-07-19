import csv
import numpy as np
import cv2


def resize_and_crop(image, img_size):
    """ Resize an image to the given img_size by first rescaling it
        and then applying a central crop to fit the given dimension. """
    source_size = np.array(image.shape[:2], dtype=float)
    target_size = np.array(img_size, dtype=float)

    # Scale
    scale = np.amax(target_size / source_size)
    inter_size = np.round(source_size * scale).astype(int)
    image = cv2.resize(image, (inter_size[1], inter_size[0]))

    # Central crop
    pad = np.round((source_size * scale - target_size) / 2.).astype(int)
    image = image[pad[0]:(pad[0] + int(target_size[0])),
                    pad[1]:(pad[1] + int(target_size[1])), :]
    
    return image


def read_timestamps(text_file):
    """
    Read a text file containing the timestamps of images
    and return a dictionary matching the name of the image
    to its timestamp.
    """
    timestamps = {'name': [], 'date': [], 'hour': [],
                  'minute': [], 'time': []}
    with open(text_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            timestamps['name'].append(row[0])
            timestamps['date'].append(row[1])
            hour = int(row[2])
            timestamps['hour'].append(hour)
            minute = int(row[3])
            timestamps['minute'].append(minute)
            timestamps['time'].append(hour + minute / 60.)
    return timestamps


def ascii_to_string(s):
    """ Convert the array s of ascii values into the corresponding string. """
    return ''.join(chr(i) for i in s)