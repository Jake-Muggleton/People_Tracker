from __future__ import division

import torch
import numpy as np
import cv2


def unique(tensor):
    """
    Returns all unique values in tensor
    """

    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """

    # get bbox coords
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get coordinates of intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # calculate intersection area
    inter_area = torch.clamp(
        inter_rect_x2 - inter_rect_x1 + 1,
        min=0
    ) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1,
        min=0
    )

    # calculate union area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
    union_area = b1_area + b2_area - inter_area

    # return IoU
    return (inter_area / union_area)


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=False):
    """
    Transforms all predictions across the 3 scales into a consistent siz
    """
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    # transform predictions to 2d
    prediction = prediction.view(
        batch_size, bbox_attrs * num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(
        batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    # adjust anchor box sizes for detection layer sizes
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # Sigmoid the coords and confidence scores
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add grid offsets to center coords prediction
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(
        1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset

    # apply anchors to dimensions of bounding box
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(
        prediction[:, :, 2:4])*anchors  # log space transforms

    # apply sigmoid to class scores
    prediction[:, :, 5:5 +
               num_classes] = torch.sigmoid((prediction[:, :, 5:5+num_classes]))

    # resize detectoin map to the size of the input image
    prediction[:, :, :4] *= stride

    return prediction


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    """
    Take predictions, prepare them, and then perform NMS on them
    """

    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask  # if below confidence threshold, set to 0

    # create new tensor of same shape and value as prediction
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2]/2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3]/2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2]/2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3]/2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)
    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]  # individual image tensor

        # reduce to max class only instead of all classes
        # index and value, max along  dim 1 of (0,1) (max class)
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5+num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        # bounding box, class, class confidence
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        # remove zeroed bounding boxes from before
        non_zero_index = (torch.nonzero(image_pred[:, 4]))
        try:
            image_pred_ = image_pred[non_zero_index.squeeze(), :].view(-1, 7)
        except:
            continue  # if no detections happen

        if image_pred_.shape[0] == 0:
            continue  # if no detections happen

        # get various classes that were detected
        # -1 holds class index, find all unique class indexes
        img_classes = unique(image_pred_[:, -1])

        # perform NMS classwise
        for c in img_classes:  # for every class present in the detections

            # get the detections of the current class
            cls_mask = image_pred_ * \
                (image_pred_[:, -1] == c).float().unsqueeze(1)
            class_mask_index = torch.nonzero(cls_mask[:, -2]).squeeze()
            image_pred_class = image_pred_[class_mask_index].view(-1, 7)

            # sort detections such that the max objectness score is at the top
            conf_sort_index = torch.sort(
                image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)  # num of detections

            # NMS
            for i in range(idx):
                # get all IoUs of the boxes after the current one in the loop
                # since we sorted bboxes by class confidence, any box with an index after the current
                # will have a smaller class confidence score
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(
                        0), image_pred_class[i+1:])
                except ValueError:
                    break
                except IndexError:
                    break

                # zero all detections with IoU > threshold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask

                # remove zero entries
                # here a zero entry is a bounding box that has too high an IoU with the current
                # bbox and since it is atleast i+1 indexed, it will have a lower confidence score
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            # write results
            batch_ind = image_pred_class.new(
                image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

    # try to return something, otherwise return a 0
    try:
        return output
    except Exception:
        return 0


def load_classes(namesFile):
    """
    Return a list of names from a class file (classnames separated by \n)
    """

    fp = open(namesFile, "r")
    names = fp.read().split("\n")[:-1]
    return names


def letterbox_image(img, inp_dim):
    """
    Resize image with unchanged aspect ratio using padding
    """

    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_img = cv2.resize(
        img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # fill a new array of size x, y, 3 with 128 values
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[
        (h-new_h)//2:(h-new_h)//2 + new_h,
        (w-new_w)//2:(w-new_w)//2 + new_w,
        :
    ] = resized_img

    return canvas


def prep_image(img, inp_dim):
    """
    Convert image from openCV format to Pytorch format
    """

    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()
    img = torch.from_numpy(img).float().div(
        255.0).unsqueeze(0)  # convert to torch and normalise
    return img

def setController(buzz, pos, ser):
    """
    Sets servo angle and buzzer state on serial port ser
    """
    s = "{0:03d}{1}".format(int(pos), buzz) # Add trailing zeros to make 3 digit number
    print(s)
    b = bytes(s, 'utf-8')

    ser.write(b) # Convert the decimal number to ASCII then send it to the Arduino
    #ser.readline() # block until arduino responds
    ser.flushInput()
    ser.flushOutput()
