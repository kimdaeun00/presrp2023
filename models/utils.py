import numpy as np 
import cv2 

def coco_dict(label_path="/workspace/models/coco_labels.txt"):
    f = open(label_path, 'r')
    lines = f.readlines()
    return [x.strip() for x in lines]


def save_image(output_path, predictions, input_img, conf_thres=0.2):
    coco = coco_dict()
    # draw bounding boxes
    labels = predictions[0]["labels"].cpu().detach().numpy()
    scores = predictions[0]["scores"].cpu().detach().numpy()
    boxes = predictions[0]["boxes"].cpu().detach().numpy()
    img = np.array(input_img)
    img = np.transpose(img, (1, 2, 0))

    color = (0, 255, 0)
    text_color = (0, 0, 0)
    for l, s, b in zip(labels, scores, boxes):
        if s<conf_thres:
            continue
        x1, y1, x2, y2 = map(int, b)
        
        # For bounding box
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # For the text background
        # Finds space required by the text so that we can put a background with that amount of width.
        (w, h), _ = cv2.getTextSize(coco[l-1], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # Prints the text.    
        img = cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        img = cv2.putText(img, coco[l-1], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

    cv2.imwrite(output_path, img)
    return