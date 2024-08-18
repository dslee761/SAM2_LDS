import cv2
import os
import json
import numpy as np
from segment_anything import sam_model_registry
from segment_anything import SamPredictor
import torch
import distinctipy
from rich.progress import track

def predict_sam(sam:SamPredictor, image_pil, boxes):
    image_array = np.asarray(image_pil)
    sam.set_image(image_array)
    transformed_boxes = sam.transform.apply_boxes_torch(boxes, image_array.shape[:2])
    masks, _, _ = sam.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(sam.device),
        multimask_output=False,
    )
    return masks.cpu()
def createDir(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print(f'Error. Failed to create the directory:{dir}')

PATH_DATA = f'/data/'
NAME_PROJECT = f'LA-POC'
PATH_IMAGE = f'{PATH_DATA}detect/Tracking/YOLO/{NAME_PROJECT}/frame/'
PATH_LABEL = f'{PATH_DATA}detect/Tracking/YOLO/{NAME_PROJECT}/result/'
PATH_WEIGHT = f'{PATH_DATA}weights/'
PATH_RESULT = f'/media/ves/rdata/detect/SAM/'
PATH_RESULT = f'{PATH_DATA}detect/Tracking/YOLO/{NAME_PROJECT}/seg/'
PATH_DRAW = f'{PATH_DATA}detect/Tracking/YOLO/{NAME_PROJECT}/draw_seg/'
NAME_MODEL = f'sam_vit_h_4b8939'
# createDir(f'{PATH_RESULT}{NAME_PROJECT}/')
# createDir(f'{PATH_RESULT}{NAME_PROJECT}/images/')
# createDir(f'{PATH_RESULT}{NAME_PROJECT}/labels/')

COLORS_PALLETE_SIZE = 64
COLORS = distinctipy.get_colors(COLORS_PALLETE_SIZE)

sam_type = "vit_h"
sam = sam_model_registry[sam_type](f'{PATH_WEIGHT}{NAME_MODEL}.pth')
sam = SamPredictor(sam)
image_list = os.listdir(f'{PATH_IMAGE}')
image_list.sort()

print(image_list)
# for image_name in image_list[0:]:
# image_name = image_list[0]
for image_name in image_list[:1]:
    if os.path.isfile(f'{PATH_DATA}detect/Tracking/YOLO/{NAME_PROJECT}/seg/{image_name.replace(".jpg", ".txt")}'):
        print(f'PASSED : {image_name}')
        continue
    image = cv2.imread(f'{PATH_IMAGE}{image_name}')
    H, W, _ = image.shape
    with open(f'{PATH_LABEL}{image_name.replace(".jpg", ".json")}', 'r') as f:
        label_json = json.load(f)
    print(label_json)
    bboxs=[]
    for label in label_json["bboxs"]:
        cx = label[0]
        cy = label[1]
        w = label[2]
        h = label[3]
        bbox = np.array(((cx - w / 2),  (cy - h / 2),  (cx + w / 2),  (cy + h / 2)), dtype=int).tolist()
        # bbox = np.array(label).tolist()
        bboxs.append(bbox)
        # cv2.rectangle(image, bbox[:2], bbox[2:], (255, 255, 0), 2)
        # print(bbox)
    # cv2.imwrite(f'{PATH_RESULT}{image_name}',image)
    print(bboxs)
    bboxs = [bboxs[0]]
    # tensor = torch.from_numpy(array) 
    bboxs = torch.tensor(bboxs, dtype=torch.float)
    # print(bboxs)
    if len(bboxs) > 0:
        masks = predict_sam(sam, image, bboxs)
        masks = masks.squeeze(1)
        if len(masks) == 0:
            print(f"No objects of the 'bbox' prompt detected in the image.")
        else:
            # Convert masks to numpy arrays
            # masks_np = [mask.squeeze().cpu().numpy() for mask in masks]
            idx = 0
            segmentations = []
            for mask in track(masks, description=f'{image_name}: {len(masks)} annotation{"s" if len(masks)>1 else ""}'):
                color = (int(COLORS[idx][2] * 255), int(COLORS[idx][1] * 255), int(COLORS[idx][0] * 255))
                mask_image = np.zeros((H, W, 3))
                for row in range(W):
                    for col in range(H):
                        mask_image[col][row] = 0 if mask[col][row] else 255
                # cv2.imwrite(f'/home/ves/label_create_sam/masks/mask_{idx:02}.jpg', mask_image)
                # mask_image = cv2.imread(f'/home/ves/label_create_sam/masks/mask_{idx:02}.jpg')
                gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
                ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                binary = cv2.bitwise_not(binary)
                contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                # for contour in contours:
                #     print(contour)
                cv2.drawContours(mask_image, contours, -1, (255, 255, 0), 2)
                cv2.drawContours(image, contours, -1, color, 2)
                segmentations.append(contours)
                # cv2.imwrite(f'/home/ves/label_create_sam/masks/mask_{idx:02}.jpg', mask_image)
                idx += 1
            cv2.imwrite(f'{PATH_RESULT}{NAME_PROJECT}/images/{image_name}', image)
            with open(f'{PATH_RESULT}{NAME_PROJECT}/labels/{image_name.replace(".jpg", ".txt")}', 'w') as f:
                for segmentation in segmentations:
                    for mask in segmentation:
                        segmentation = np.reshape(mask, -1).tolist()
                        for point in mask:
                            f.write(f'{point} ')
                    f.write(f'\n')
