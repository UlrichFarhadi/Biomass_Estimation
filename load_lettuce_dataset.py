#---------- Imports ----------
import numpy as np
from PIL import Image
import glob
import matplotlib as mpl
import re
import json
from tqdm import tqdm

def load_all_images():
    with open("Dataset/result.json") as f:
        data = f.read()
        rgb_list = []
        depth_list = []
        fresh_weight_list = []
        dry_weight_list = []
        for filename in tqdm(glob.glob('Dataset/Debth*.png')): #assuming gif
            num = re.findall(r'\d+',filename )

            img_depth = Image.open("Dataset/Debth_" + num[0] +".png") #Image.open("Dataset/Debth_1.png")
            img_depth = (img_depth / np.linalg.norm(img_depth))*255
            cm = mpl.cm.get_cmap('jet')
            img_depth = cm(np.array(img_depth))
            img_depth = Image.fromarray(np.uint8(img_depth[:,:,:3]*255))

            img_rgb = Image.open("Dataset/RGB_" + num[0] +".png")
            js = json.loads(data)

            FreshWeight = js.get( num[0]).get("FreshWeightShoot")
            DryWeight = js.get( num[0]).get("DryWeightShoot")

            rgb_list.append(img_rgb)
            depth_list.append(img_depth)
            fresh_weight_list.append(FreshWeight)
            dry_weight_list.append(DryWeight)
    return rgb_list, depth_list, fresh_weight_list, dry_weight_list