'''
Testing image properties to transform day to night
-- Lower exposure
-- Lower temperature (more blue)

'''

from PIL import Image 
#from deprecated_scipy import toimage
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure as skie


kelvin_table = {
    1000: (255,56,0),
    1500: (255,109,0),
    2000: (255,137,18),
    2500: (255,161,72),
    3000: (255,180,107),
    3500: (255,196,137),
    4000: (255,209,163),
    4500: (255,219,186),
    5000: (255,228,206),
    5500: (255,236,224),
    6000: (255,243,239),
    6500: (255,249,253),
    7000: (245,243,255),
    7500: (235,238,255),
    8000: (227,233,255),
    8500: (220,229,255),
    9000: (214,225,255),
    9500: (208,222,255),
    10000: (204,219,255)}

def convert_temp(image, temp):
    r, g, b = kelvin_table[temp]
    matrix = ( r / 255.0, 0.0, 0.0, 0.0,
               0.0, g / 255.0, 0.0, 0.0,
               0.0, 0.0, b / 255.0, 0.0 )
    return image.convert('RGB', matrix)

def to_night(image):


    #blue temperature
    img_night = convert_temp(image, 9500)

    return img_night


def show(img):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img, cmap=plt.cm.gray)
        ax.set_axis_off()
        plt.show()


if __name__ == "__main__":

    '''
    Try to variate image temperature to blue-ish + lower exposure
    '''
    #im_src = Image.open("our_demo_images/source2.jpg").convert('RGB')
    #img_night = to_night(im_src)
    #toimage(img_night, cmin=0.0, cmax=255.0).save('test.png')

    '''
    Matching histograms between target image (at night) and source image (daylight)
    '''

    img = plt.imread("source.jpg")
    img2 = plt.imread("source2.jpg")
    target = plt.imread("night_target.jpeg")

    new_img = skie.match_histograms(img, target, channel_axis = 2)
    new_img2 = skie.match_histograms(img2, target, channel_axis = 2)
    show(new_img)
    show(new_img2)
    