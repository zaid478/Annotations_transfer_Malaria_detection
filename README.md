# Annotations_transfer_Malaria_detection
This repository contains code to map patches from 1000x (captured from mobile camera attached to microscope) to 400x and 100x

## Finding 1000x patches in 400x images
python multiscale_template_1000_400.py --template_folder /path/to/1000x/images --image_folder /path/to/400x/frames --num_images number --root_folder /path/to/root 


## Finding 1000x patches in 100x images
python multiscale_template_1000_100.py --template_folder /path/to/1000x/images --image_folder /path/to/100x/images --num_images number --root_folder /path/to/root 


### Arguments
template_folder: This should be the folder where the patches of 1000x are stored <br/>
image_folder: This should be the folder where the images (either 400x or 100x) are stored. Keep in mind that 400x images are usually created from video, while for 100x, there is no video used; Direct images should be captured using microscope. <br/>
num_images: The number of images in the folder image_folder <br/>
root_folder: The outer folder in which template_folder and image_folder is resided <br/>


## Directory structure
root_folder <br/>
  -- template_folder <br/>
  -- image_folder <br/>
