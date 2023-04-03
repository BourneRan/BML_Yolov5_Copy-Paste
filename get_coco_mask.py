# """
# get semantic segmentation annotations from coco data set.
# """
# from PIL import Image
# import imgviz
# import argparse
# import os
# import tqdm
# from pycocotools.coco import COCO

 
# def save_colored_mask(mask, save_path):
#     lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
#     colormap = imgviz.label_colormap()
#     lbl_pil.putpalette(colormap.flatten())
#     lbl_pil.save(save_path)
 
 
# def main(args):
#     annotation_file = os.path.join(args.input_dir, 'trainval.json')
#     os.makedirs(os.path.join(args.input_dir, 'SegmentationClass'), exist_ok=True)
#     os.makedirs(os.path.join(args.input_dir, 'JPEGImages'), exist_ok=True)
#     coco = COCO(annotation_file)
#     catIds = coco.getCatIds()
#     imgIds = coco.getImgIds()
#     print("catIds len:{}, imgIds len:{}".format(len(catIds), len(imgIds)))
#     for imgId in tqdm.tqdm(imgIds, ncols=100):
#         img = coco.loadImgs(imgId)[0]
#         annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
#         anns = coco.loadAnns(annIds)
#         if len(annIds) > 0:
#             mask = coco.annToMask(anns[0]) * anns[0]['category_id']
#             for i in range(len(anns) - 1):
#                 mask += coco.annToMask(anns[i + 1]) * anns[i + 1]['category_id']
#             img_origin_path = os.path.join(args.input_dir, 'BML250', img['file_name'])
#             img_output_path = os.path.join(args.input_dir, 'JPEGImages', img['file_name'])
#             seg_output_path = os.path.join(args.input_dir, 'SegmentationClass', img['file_name'].replace('.bmp', '.png'))
#             shutil.copy(img_origin_path, img_output_path)
#             save_colored_mask(mask, seg_output_path)
 
 
# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_dir", default="BML", type=str,
#                         help="coco dataset directory")
#     parser.add_argument("--split", default="", type=str,
#                         help="train2017 or val2017")
#     return parser.parse_args()
 
 
# if __name__ == '__main__':
#     args = get_args()
#     main(args)

import numpy as np
import imgviz
import argparse
import os
import tqdm
import shutil
from PIL import Image
from pycocotools.coco import COCO


def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)


def points_to_mask(points, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = points
    mask[int(y1):int(y2), int(x1):int(x2)] = 1
    return mask


def main(args):
    annotation_file = os.path.join(args.input_dir, 'trainval.json')
    os.makedirs(os.path.join(args.input_dir, 'SegmentationClass'), exist_ok=True)
    os.makedirs(os.path.join(args.input_dir, 'JPEGImages'), exist_ok=True)
    coco = COCO(annotation_file)
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds()
    print("catIds len:{}, imgIds len:{}".format(len(catIds), len(imgIds)))
    for imgId in tqdm.tqdm(imgIds, ncols=100):
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        if len(annIds) > 0:
            points = anns[0]['segmentation'][0]
            mask = points_to_mask(points, img['height'], img['width']) * (anns[0]['category_id'] + 1)

            for i in range(len(anns) - 1):
                points = anns[i + 1]['segmentation'][0]
                mask += points_to_mask(points, img['height'], img['width']) * (anns[i + 1]['category_id'] + 1)
            img_origin_path = os.path.join(args.input_dir, 'BML250', img['file_name'])
            img_output_path = os.path.join(args.input_dir, 'JPEGImages', img['file_name'])
            seg_output_path = os.path.join(args.input_dir, 'SegmentationClass', img['file_name'].replace('.bmp', '.bmp'))
            shutil.copy(img_origin_path, img_output_path)
            save_colored_mask(mask, seg_output_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="BML", type=str,
                        help="coco dataset directory")
    parser.add_argument("--split", default="", type=str,
                        help="train2017 or val2017")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
