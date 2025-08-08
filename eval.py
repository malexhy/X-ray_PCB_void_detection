import numpy as np
import cv2
import glob
import torch
from torchvision import transforms
import segment_model as seg
import time
import os
from prettytable import PrettyTable
import csv
import argparse


INPUT_SHAPE = (1, 256, 256)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def processing_pred(img, pred):
    ICMask = pred[0]
    bubbleMask = pred[1]
    ICMask = np.where(ICMask>0.5, np.ones(ICMask.shape), 0).astype('uint8')
    ICMask = cv2.morphologyEx(ICMask, cv2.MORPH_OPEN, np.ones((10,10)))
    ICMask = cv2.morphologyEx(ICMask, cv2.MORPH_CLOSE, np.ones((10,10)))
    bubbleMask = np.where(bubbleMask>0.5, np.ones(bubbleMask.shape), 0)

    ic_color = np.array([255,0,0], dtype='uint8')
    bubble_color = np.array([0,0,255], dtype='uint8')
    both_color = np.array([0,255,0], dtype='uint8')
    
    ori_im = (cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)*255).astype('uint8')

    # mask = np.where(ICMask[...,None], ic_color, ori_im)
    mask = np.where(bubbleMask[...,None], bubble_color, ori_im)
    
    mask = cv2.addWeighted(ori_im, 0.5, mask, 0.5, 0)

    contours, _ = cv2.findContours(ICMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)


    for j, contour in enumerate(contours):

        if cv2.contourArea(contour) < (100*100):
            break
        rect = cv2.boundingRect(contour)
        x,y,w,h = rect

        bubble_pixel = bubbleMask[y:y+h, x:x+w].sum()
        ic_area = w*h
        persentage = (bubble_pixel/ic_area)*100
        persentage = '{}'.format(persentage)[:5] + '%'

        
        cv2.rectangle(mask, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.rectangle(mask, (x,y-40), (x+130,y), (0,255,0), -1)
        cv2.putText(mask,persentage, (x+10,y-10), 1, 2, (0,0,255), 2)

    return mask


def oom_pred(x, model, device):
    dev = 4
    size = torch.as_tensor(x.size()[-2:]).to(torch.float)
    new_size = (((size/dev)/32).to(torch.int)+1)*32
    skip = ((size-new_size)/(dev-1)).to(torch.int)
    pred = torch.zeros((1, 2, int(size[0].item()), int(size[1].item())), dtype=torch.float).to(device)
    pred_mask = torch.zeros((1, 2, int(size[0].item()), int(size[1].item())), dtype=torch.float).to(device)
    for i in range(dev):
        for j in range(dev):
            start_i = i*skip[0] if i < (dev-1) else size[0].to(torch.int) - new_size[0]
            start_j = j*skip[1] if j < (dev-1) else size[1].to(torch.int) - new_size[1]
            end_i = start_i + new_size[0] if i < (dev-1) else size[0].to(torch.int) 
            end_j = start_j + new_size[1] if j < (dev-1) else size[1].to(torch.int) 
            x_temp = x[..., start_i:end_i, start_j:end_j]
            pred_temp = model(x_temp)
            pred[..., start_i:end_i, start_j:end_j] += pred_temp
            pred_mask[..., start_i:end_i, start_j:end_j] += torch.ones(pred_temp.shape, dtype=torch.float).to(device)
    pred = pred/pred_mask
    return pred

def predict_images_folder(model, images_dir, save_dir, device, train_data=None):
    oom = False
    transformer = transforms.Compose([  transforms.ToPILImage(),
                                        transforms.RandomCrop(size=(1536 ,1952), padding=[4,4], padding_mode='edge'),
                                        transforms.ToTensor()])

    print("Loading images...")
    images_path = [x for x in glob.glob(images_dir + '/*.jpg') if "BubbleMask" not in x and "ICMask" not in x]
    print("Find {} image in '{}'".format(len(images_path), images_dir))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()
    for i, image_path in enumerate(images_path):
        startime = time.time()
        image_name = image_path.replace('\\', '/').split('/')[-1].split('.')[0]
        if train_data is not None:
            if image_name in train_data:
                continue
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        tensor_img = transformer(image)
        with torch.no_grad():
            if oom:
                pred = oom_pred(tensor_img[None,...].to(device), model, device)
            else: 
                try:
                    pred = model(tensor_img[None,...].to(device))
                except RuntimeError:
                    oom = True
                    pred = oom_pred(tensor_img[None,...].to(device), model, device)    
        pred = pred.cpu().detach().numpy()[0]
        mask = processing_pred(tensor_img.cpu().detach().numpy()[0], pred)

        save_path = save_dir + '/' + image_name + '_result.jpg'
        cv2.imwrite(save_path, mask)
        endtime = time.time()
        print("image: {},\ttime: {:.4F}".format(image_name, endtime-startime))

def generate_result(weight_path, images_dir=None, save_dir=None, train_dir=None, device='cuda'):
    model_name = weight_path.split('/')[-2].split('_')[:2]
    
    weights = glob.glob(weight_path + '/*')
    best_weight = sorted(weights)[-1].replace('\\', '/')
    if save_dir is None:
        return best_weight
    print("Creating Model...")
    model = seg.create_model(model_name[0], model_name[1], INPUT_SHAPE, 2, 'sigmoid', device).model
    print("Model name: {}, Parms: {}".format(model_name, count_parameters(model)))
    print("Loading weight...")
    print(model.load_state_dict(torch.load(best_weight)))
    temp = best_weight.split('/')
    train_data_name = '_'.join(weight_path.split('/')[-2].split('_')[2:-2]).split("2and")[-1]
    train_data_path = train_dir + '/' + train_data_name + '/train'
    train_data = glob.glob(train_data_path + '/*_ICMask.jpg')
    train_data = list(map(lambda x: x.replace('\\', '/').split('/')[-1].replace('_ICMask.jpg', ''),
                        train_data))

    save_dir = save_dir + '/' + temp[-3] + '_' + temp[-1]
    predict_images_folder(model, images_dir, save_dir, device, train_data)

def get_best_result_from_dir(log_dir, device):
    best_weight = None
    best_result = 0
    for weight_path in glob.glob(log_dir + '/*/Weight'):
        weight_path = weight_path.replace('\\','/')
        weight = generate_result(weight_path, device)
        result = weight.split('/')[-1].split('_')[-1]
        result = float(result)/100
        if result > best_result:
            best_result = result
            best_weight = weight
    return best_weight

def print_result_differemt_model(log_dir):
    log_path = glob.glob(log_dir + '/*')
    result = {}

    for log in log_path:
        log_info = log.replace('\\', '/').split('/')[-1].split('_')
        encoder = log_info[0]
        decoder = log_info[1]
        best_result = sorted(glob.glob(log + '/Weight/*'))[-1].replace('\\', '/').split('/')[-1].split('_')[-1]
        best_result = float(best_result)/100
        if encoder not in result.keys():
            result[encoder] = {}
        result[encoder][decoder] = best_result

    max_key = []
    for encoder in result:
        for key in result[encoder].keys():
            if key not in max_key:
                max_key.append(key)
    max_key = sorted(max_key)
    temp_key = max_key.copy()
    temp_key.insert(0, 'Encoder\\Decoder')
    tab = PrettyTable()
    tab.field_names = temp_key
    for encoder in result:
        temp = [encoder]
        for key in max_key:
            temp.append(result[encoder][key] if key in result[encoder].keys() else '/')
        tab.add_rows([temp])
    tab.align[temp_key[0]]='l'
    tab.sortby = 'UnetPlusPlus'
    print(tab)
    return tab


def save_csv(save_dir, result):
    result = [a.split(',') for a in result.get_csv_string().split('\n')]
    # print(result)
    with open(save_dir, 'w', newline='') as outcsv:
        writer = csv.writer(outcsv)
        writer.writerows(result)

def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    images_dir = args.test_path
    save_dir = args.save_path
    log_dir = args.log_path
    dataset_root = args.dataset_root
    
    if not os.path.exists(log_dir):
        raise ValueError(f'Cannot find path {log_dir}')
    
    result_name = log_dir.split('/')[-1]
    save_dir = save_dir + '/' + result_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ious = print_result_differemt_model(log_dir)
    save_csv(f'{save_dir}/IOU.csv', ious)
    print(f'Saving result to {save_dir} ......')
    
    
    print(f'Generating result ......')

    weight_path = '/'.join(get_best_result_from_dir(log_dir, device).split('/')[:-1])
    ds_name = weight_path.split('/')[-2].split('_')[-2]
    generate_result(weight_path, images_dir, save_dir + '/image', dataset_root + '/' + ds_name, device)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("XrayPCB")
    
    
    parser.add_argument("--log_path", type=str, default='./Run/09_08_2025_01_56_39', help="Path saved models and logs")
    parser.add_argument("--test_path", type=str, default='./test_data', help="Path to test data folder")
    parser.add_argument("--save_path", type=str, default='./Results', help="Path to save the result")
    parser.add_argument("--dataset_root", type=str, default='./datasets', help="Path to the train data folder root")

    
    
    
    args = parser.parse_args()
    
    main(args)
    