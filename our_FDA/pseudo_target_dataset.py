import numpy as np
import torch
from PIL import Image, ImageEnhance
from utils import FDA_source_to_target
from deprecated_scipy import toimage
import random
from glob import glob
import skimage.exposure as skie

def create_pseudo_target_image(src_img, trg_img, output_dir, src_img_path):

    im_src = np.asarray(src_img, np.float32)
    im_trg = np.asarray(trg_img, np.float32)

    im_src = im_src.transpose((2, 0, 1))
    im_trg = im_trg.transpose((2, 0, 1))

    im_src = torch.from_numpy(im_src).unsqueeze(0)
    im_trg = torch.from_numpy(im_trg).unsqueeze(0)

    src_in_trg = FDA_source_to_target( im_src, im_trg, L=0.01 )

    src_in_trg = torch.Tensor.numpy(src_in_trg.squeeze(0))

    src_in_trg = src_in_trg.transpose((1,2,0))
    src_in_trg = src_in_trg.astype(int)

    label = src_img_path.split('/')[-1]
    toimage(src_in_trg, cmin=0.0, cmax=255.0).save(f'{output_dir}/{label}')


def create_pseudo_target_dataset(source_dir, target_dir, output_dir):

    source_paths = glob(f"{source_dir}/*.jpg", recursive=True)
    targets_paths = glob(f"{target_dir}/**/*.jpg", recursive=True)

    for src_img_path in source_paths:
        #Get a random number between 0 and 4
        index = int(random.uniform(0, len(targets_paths)))
        print(f'source: {source_dir} & target: {targets_paths[index]}')
        src_img = Image.open(src_img_path).convert('RGB')
        trg_img = Image.open(targets_paths[index]).convert('RGB')
        
        create_pseudo_target_image(src_img, trg_img, output_dir, src_img_path)

    return

def matchhistograms(src_img):
    trg_img = Image.open("target_test/target2.jpg")
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

'''
FDA is not enough to achieve good night images, so tried some post processing techniques used by photographers:
-- Match histograms with target's one
-- Reduce Saturation to 50%
-- Add blue tint
-- Reduce overall brightness (?)
'''
def better_night_images(input_dir, output_dir):
    
    input_paths = glob(f"{input_dir}/*.jpg", recursive=True)

    for img_path in input_paths:
        img = Image.open(img_path).convert('RGB')
        img = matchhistograms(img)
        #lower saturation
        saturation = ImageEnhance.Color(img)
        img = saturation.enhance(0.6)
        #blue tint
        img = convert_temp(img, 10000)

        # [... TO DO: blue tint, gradient]
        img_path = img_path.split("/")[-1]
        toimage(img, cmin=0.0, cmax=255.0).save(f'{output_dir}/{img_path}')



if __name__ == '__main__':

    BETTER_NIGHT = False
    DEBUG = False

    source_dir = "source_test"
    target_dir = "target_test"
    output_dir = "out_test"

    if DEBUG == False:
        source_dir = "/content/small/test/queries_v1"
        target_dir = "/content/tokyo_night/test/queries_v1"
        output_dir = "/content/queries_v1_pseudotarget"

    create_pseudo_target_dataset(source_dir, target_dir, output_dir)
    
    if BETTER_NIGHT == True:
        better_night_images(output_dir, output_dir)






