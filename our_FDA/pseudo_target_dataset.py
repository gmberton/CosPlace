import os
import numpy as np
from PIL import Image, ImageEnhance
from utils import FDA_source_to_target_np
from deprecated_scipy import toimage
import random
from glob import glob
import skimage.exposure as skie

def create_pseudo_target_image(src_img, trg_img, output_path):

    size = src_img.size

    im_src = src_img.resize( (1024,512), Image.Resampling.BICUBIC )
    im_trg = trg_img.resize( (1024,512), Image.Resampling.BICUBIC )

    im_src = np.asarray(im_src, np.float32)
    im_trg = np.asarray(im_trg, np.float32)

    im_src = im_src.transpose((2, 0, 1))
    im_trg = im_trg.transpose((2, 0, 1))

    src_in_trg = FDA_source_to_target_np( im_src, im_trg, L=0.0025 )

    src_in_trg = src_in_trg.transpose((1,2,0))

    #Images will be saved with a NIGHT at the beginning, so we can use this later for the domain classification task

    #it's ok put an additional string at the beginning, because field 1 is UTM east, field 2 is UTM north, field 9 is heading
    #and field 0 is not being used when calling .split('@')
    output_path = output_path.replace('@', 'NIGHT@', 1)
    toimage(src_in_trg, cmin=0.0, cmax=255.0).resize(size).save(output_path)
    return output_path

'''
Transform the source_dir (in our case small/train and small/val/queries_v1) into pseudo-target,
by using 5 images from target_dir (in our case target folder containing 5 queries from tokyo-night)

All images are saved with a 'night' tag at the end: ../../@something@something...@night@.jpg

Default steps = 1 --> all the images will be transformed
= 2 --> Only half of them
= 3 --> A third
        . . . and so on
'''
def create_pseudo_target_dataset(source_dir, target_dir, steps = 1):

    targets_paths = glob(f"{target_dir}/**/*.jpg", recursive=True)

    for root, dirs, files in os.walk(source_dir, topdown=True):
        for i in range(0, len(files), steps): 
            if '.DS_Store' in files[i]:
                continue

            print(f'transforming... {os.path.join(root, files[i])}')
            #Get a random photos between the 5 from target dataset
            index = int(random.uniform(0, len(targets_paths)))
            src_img = Image.open(os.path.join(root, files[i])).convert('RGB')
            trg_img = Image.open(targets_paths[index]).convert('RGB')
            out_path = create_pseudo_target_image(src_img, trg_img, output_path=os.path.join(root, files[i]))
            src_img = Image.open(out_path).convert('RGB')
            better_night_images(src_img, output_path=out_path)

'''
FDA is not enough to achieve good night images, so tried some post processing techniques used by photographers:
-- Match histograms with target's one (bad image results --> scraped)
-- Reduce Saturation to 70%
-- Change image temperature to a more blue-ish one
'''

def matchhistograms(src_img):
    trg_img = Image.open("target/@0381904.46@3946329.36@54@S@035.65375@0139.69540@00399.jpg@@@@@@@@.jpg")
    trg_img = np.array(trg_img)
    na = np.array(src_img)
    matched = skie.match_histograms(na, trg_img, channel_axis = 2)
    #toimage(matched, cmin=0.0, cmax=255.0).save(f'{src_img_path}')
    return Image.fromarray(matched)

def convert_temp(image):
    r, g, b = (180,219,255)
    matrix = ( r / 255.0, 0.0, 0.0, 0.0,
               0.0, g / 255.0, 0.0, 0.0,
               0.0, 0.0, b / 255.0, 0.0 )
    return image.convert('RGB', matrix)

def better_night_images(input_img, output_path):

        img = input_img
    
        #img = matchhistograms(img)
        #lower saturation
        saturation = ImageEnhance.Color(img)
        img = saturation.enhance(0.7)
        #blue tint
        img = convert_temp(img)
        toimage(img, cmin=0.0, cmax=255.0).save(output_path)

        return img
        


'''
This file is meant to be used offline to produce pseudo-target images of
- '/content/small/train'
- '/content/small/val/queries_v1
from SF-XS as source dataset in a Google Colab environment.

Please enter the correct paths if they're different.

The output will be added to the old dataset path. So in small/train you will
find both day images and night images.
'''
if __name__ == '__main__':
    
    source_dir = "/content/small/train"
    target_dir = "/content/AML23-CosPlace/our_FDA/target"
    num_steps = 3

    create_pseudo_target_dataset(source_dir, target_dir, steps=num_steps)





