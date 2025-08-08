import torch
from torchsummary import summary
import time
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import cv2
import numpy as np
from datetime import datetime
import segmentation_models_pytorch as smp
import os
import copy


class Model():
    def __init__(self, model, 
                input_shape=(1, 256, 256),  
                device = "cuda", 
                name = "segNet",
                do_summary=False):
        self.oom = False
        self.model = model
        self.device = device
        self.name = name
        self.model.to(device)
        self.loss_fn = None
        self.metrics = None
        if do_summary:
            summary(self.model, input_shape, device=device)



    def fit(self, train_data, loss_fn, opt, lr=0.001, metrics=[], test_data=None, epoch=0, log_image=False, save_path=None, comment='_'):
        self.loss_fn = loss_fn
        self.opt = opt(self.model.parameters(), lr)
        self.metrics = metrics

        save_path = save_path if save_path is not None else './Weight/{}'.format(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
        self.writer = SummaryWriter(save_path + '/log')
        save_path_weight = save_path + '/Weight/'
        if not os.path.exists(save_path_weight):
            os.makedirs(save_path_weight)
        save_path_weight = save_path_weight + comment
        best_test_IOU = 0
        best_epoch = 0
        best_weight = [copy.deepcopy(self.model.state_dict())]
        for i in range(epoch):
            print('Epoch: {}'.format(i+1), end='\t\t')
            testIOU = self.training_loop(i, train_data, test_data, log_image=log_image)
            print('')

            if testIOU > best_test_IOU:
                best_test_IOU = testIOU
                best_epoch = i
                best_test_IOU_str = '{:.4F}'.format(best_test_IOU).replace('.', '_')
                ppp = best_weight.pop(0)
                del ppp
                best_weight.append(copy.deepcopy(self.model.state_dict()))

        
        self.writer.flush()
        self.writer.close()
        # torch.save(self.model, './Model/segmodel')
        torch.save(best_weight[0], save_path_weight+'epoch{}_{}'.format(best_epoch, best_test_IOU_str))
        torch.save(self.model.state_dict(), save_path_weight+'end')
        
        del best_weight[0]
        del best_weight



    def train_one_step(self, x, y):
        self.opt.zero_grad()
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        y = (y>0.3).to(torch.uint8)
        tp, fp, fn, tn = smp.metrics.get_stats(pred, y, mode='multilabel', threshold=0.5)
        results = {}
        for metric_ in self.metrics:
            results[metric_] = self.metrics[metric_](tp, fp, fn, tn, reduction='micro')
        loss.backward()
        self.opt.step()
        return loss.item(), results
    
    @torch.no_grad()
    def test_one_step(self, x, y, return_pred=False):
        if self.oom:
            pred = self.oom_pred(x, y)
        else: 
            try: 
                pred = self.model(x)
            except RuntimeError:
                self.oom = True
                pred = self.oom_pred(x, y) 

        loss = self.loss_fn(pred, y)
        y = (y>0.5).to(torch.uint8)
        tp, fp, fn, tn = smp.metrics.get_stats(pred, y, mode='multilabel', threshold=0.5)
        results = {}
        for metric_ in self.metrics:
            results[metric_] = self.metrics[metric_](tp, fp, fn, tn, reduction='micro')

        if return_pred:
            return loss.item(), results, pred
        else:
            return loss.item(), results


    def training_loop(self, epoch, train_data, test_data=None, log_image=False):
        start_time = time.time()

        trainLoss = 0
        testLoss = 0
        return_value = 0
        results_temp = {}
        

        loss_log = {}
        results_log = {}

        for key in self.metrics:
            results_temp[key] = 0
            results_log[key] = {}

        # train
        self.model.train()
        for i, (x, y) in enumerate(itr_merge(train_data)):
            (x, y) = (x.to(self.device), y.to(self.device))
            # print(x.shape)
            # exit()
            loss, results = self.train_one_step(x, y)

            trainLoss += loss
            for key in results:
                results_temp[key] += results[key].item()
        
        loss_log["train"] = trainLoss/(i+1)
        
       
        end_time = time.time()
        print('Time: {:.4f},'.format(end_time-start_time), end='\t\t')
        print('Train_Loss: {:.4f},'.format(loss_log["train"]), end='\t\t')
        for key in results:
            results_log[key]["train"] = results_temp[key]/(i+1)
            print('Train_{}: {:.4f}'.format(key, results_log[key]["train"]), end='\t\t')        


        # test
        for key in results:
            results_temp[key] = 0
        self.model.eval()
        if not isinstance(test_data, type(None)):
            if ((epoch)%10) == 0:
                
                with torch.no_grad():
                    for i, (x, y) in enumerate(itr_merge(test_data)):
                        loss = None
                        results = None
                        (x, y) = (x.to(self.device), y.to(self.device))
                        if ((epoch % 20) == 0) and (i == 0) and (log_image):
                            loss, results, pred = self.test_one_step(x, y, return_pred=True)
                            self.writer.add_figure('predictions vs. actuals', plot_pred(x, pred), global_step=epoch)
                        else:
                            loss, results = self.test_one_step(x, y)

                        testLoss += loss
                        for key in results:
                            results_temp[key] += results[key].item()
                        if i == 10:
                            break
                    
                    loss_log["test"] = testLoss/(i+1)
                    
                    
                print('Val_Loss: {:.4F}'.format(loss_log["test"]), end='\t\t')
                for key in results:
                    results_log[key]["test"] = results_temp[key]/(i+1)
                    print('Test_{}: {:.4f}'.format(key, results_log[key]["test"]), end='\t\t')   

                return_value = results_log['IOU']['test']
        self.writer.add_scalars('Loss', loss_log, epoch)
        for key in results:
            self.writer.add_scalars(key, results_log[key], epoch)
        
        return return_value

    @torch.no_grad()
    def oom_pred(self, x, y):
        dev = 4
        size = torch.as_tensor(x.size()[-2:]).to(torch.float)
        new_size = (((size/dev)/32).to(torch.int)+1)*32
        skip = ((size-new_size)/(dev-1)).to(torch.int)
        pred = torch.zeros(y.size(), dtype=torch.float).to(self.device)
        pred_mask = torch.zeros(y.size(), dtype=torch.float).to(self.device)
        for i in range(dev):
            for j in range(dev):
                start_i = i*skip[0] if i < (dev-1) else size[0].to(torch.int) - new_size[0]
                start_j = j*skip[1] if j < (dev-1) else size[1].to(torch.int) - new_size[1]
                end_i = start_i + new_size[0] if i < (dev-1) else size[0].to(torch.int) 
                end_j = start_j + new_size[1] if j < (dev-1) else size[1].to(torch.int) 
                x_temp = x[..., start_i:end_i, start_j:end_j]
                pred_temp = self.model(x_temp)
                pred[..., start_i:end_i, start_j:end_j] += pred_temp
                pred_mask[..., start_i:end_i, start_j:end_j] += torch.ones(pred_temp.shape, dtype=torch.float).to(self.device)
        pred = pred/pred_mask
        return pred



# ATTENTION_TYPE = None

        
def plot_pred(x, pred):
    ori_im = x.cpu().detach().numpy()[0][0][...,None]
    ori_im = (cv2.cvtColor(ori_im, cv2.COLOR_GRAY2RGB)*255).astype('uint8')

    pred = pred.cpu().detach().numpy()[0]
    ICMask = pred[0]
    bubbleMask = pred[1]

    ICMask = np.where(ICMask>0.5, np.ones(ICMask.shape), 0).astype('uint8')
    ICMask = cv2.morphologyEx(ICMask, cv2.MORPH_OPEN, np.ones((10,10)))
    ICMask = cv2.morphologyEx(ICMask, cv2.MORPH_CLOSE, np.ones((10,10)))
    bubbleMask = np.where(bubbleMask>0.5, np.ones(bubbleMask.shape), 0)

    bubble_color = np.array([255,0,0], dtype='uint8')
    comb_im = np.where(bubbleMask[...,None]>0.5, bubble_color, ori_im)
    comb_im = cv2.addWeighted(ori_im, 0.5, comb_im, 0.5, 0)

    contours, _ = cv2.findContours(ICMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < (200*200):
            break
        # peri = cv2.arcLength(contour, True)
        # approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
        rect = cv2.boundingRect(contour)

        x,y,w,h = rect
        bubble_inline = bubbleMask[y:y+h, x:x+w]
        areaOfBubble = bubble_inline.sum()
        persentage = areaOfBubble/(h*w)*100
        persentage = '{}'.format(persentage)[:5] + '%'

        cv2.rectangle(comb_im,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(comb_im,(x,y-40),(x+130,y),(0,255,0),-1)
        cv2.putText(comb_im,persentage,(x+10,y-10),1,2,(0,0,255), 2)
    
    fig = plt.figure()
    plt.imshow(comb_im.astype('uint8'))
    plt.axis('off')
    plt.tick_params(top='off', bottom='off', left='off', right='off')

    return fig


def create_model(encoder:str, decoder:str, input_shape, output_channels, activation, device, do_summary=False, **kwargs):
    model_name = encoder + '_' + decoder
    input_channels= input_shape[0]
    if decoder in ['Unet', 'UnetPlusPlus']:
        segmodel = smp.create_model(decoder, encoder.lower(), 
                                    encoder_weights="imagenet",
                                    in_channels=input_channels,             
                                    classes=output_channels,                
                                    activation=activation, 
                                    **kwargs
                                    )
    else:
        segmodel = smp.create_model(decoder, encoder.lower(), 
                                    encoder_weights="imagenet",
                                    in_channels=input_channels,             
                                    classes=output_channels,                
                                    activation=activation, 
                                    )
    model = Model(segmodel, input_shape, device=device, do_summary=do_summary, name=model_name)
    return model

def itr_merge(itrs):
    for itr in itrs:
        for v in itr:
            yield v
