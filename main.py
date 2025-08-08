import torch
from segment_model import create_model
from torch import nn
import segmentation_models_pytorch as smp
from datetime import datetime
import os
from data_loader import x_ray_image_dataLoader as xrid
from data_loader import x_ray_image_dataLoader_testing_set
import argparse


IMG_ORI_SIZE = {
    'd1': (1120,1792),
    'd2': (1536,1952),
}

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    input_shape = (args.in_channels, args.in_size, args.in_size)
    crop_size = (args.in_size+128, args.in_size+128)

    loss_fns = {"BCELoss": nn.BCELoss(reduction='mean'),
                "DiceLoss": smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=False),
                "JaccardLoss": smp.losses.JaccardLoss(smp.losses.MULTILABEL_MODE, from_logits=False)}

    metrics = {"IOU": smp.metrics.iou_score, "ACC": smp.metrics.accuracy}
    
    opt = torch.optim.Adam
    lr = args.lr
    only_icbubbles = True

    if args.encoder == 'all':
        encoder_list = ['efficientnet-b0', 'efficientnet-b3', "VGG13", 'Resnet18', 'Resnet34']
    elif isinstance(args.encoder, list):
        encoder_list = args.encoder
    else:
        encoder_list = [args.encoder]
        
    
    if args.decoder == 'all':
        decoder_list = ['Unet', 'UnetPlusPlus', 'FPN', 'PSPNet', 'DeepLabV3', 'PAN']
    elif isinstance(args.decoder, list):
        decoder_list = args.decoder
    else:
        decoder_list = [args.decoder]
    
    data_path_base = args.data_root
    data_path = f'{data_path_base}/{args.dataset}'
    data_name = args.dataset
    loss_name = args.loss
    l_max = args.l_max

    date_info = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    
    for encoder in encoder_list:
        for decoder in decoder_list:
            for i in range(1):
                if (encoder=='VGG13') and (decoder in ['DeepLabV3', 'PAN']):
                    continue
                torch.cuda.empty_cache()

                
                loss_fn = loss_fns[loss_name]
                train_data_path = data_path + '/train'
                test_data_path = data_path + '/test'


                print('Loading data...')

                loader = xrid(train_data_path, 3, (input_shape[1], input_shape[2]), crop_size=crop_size, shuffle=True, only_icbubbles=only_icbubbles, force_positive_loop=l_max)
                test_loader = x_ray_image_dataLoader_testing_set(test_data_path, 1, IMG_ORI_SIZE[data_name], shuffle=True, only_icbubbles=only_icbubbles)
                
                print('Done!')

                print('Creating model...')
                model = create_model(encoder, decoder, input_shape, 2, 'sigmoid', device, do_summary=False)
                print('Model name: ' + model.name)

                save_path = f'./Run/{date_info}/{model.name}_{data_name}_{loss_name}' 
                print("Save path: " + save_path)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                print('Start training...')
                model.fit([loader.loader], 
                        loss_fn, opt, 
                        lr=lr, 
                        metrics=metrics,
                        test_data=[test_loader.loader], 
                        save_path=save_path,
                        epoch=5000)
                
                print('Finish\n\n')
    
    
    
    
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("XrayPCB")
    
    parser.add_argument("--in_channels", type=int, default=1, help="Number of input channel")
    parser.add_argument("--out_channels", type=int, default=2, help="Number of output channel")
    parser.add_argument("--encoder", type=str, default='all', choices=['efficientnet-b0', 'efficientnet-b3', "VGG13", 'Resnet18', 'Resnet34', 'all'])
    parser.add_argument("--decoder", type=str, default='all', choices=['Unet', 'UnetPlusPlus', 'FPN', 'PSPNet', 'DeepLabV3', 'PAN', 'all'])

    parser.add_argument("--batch_size", type=int, default=3, help="Batch size")
    parser.add_argument("--n_epoch", type=int, default=5000, help="Number of training epoch")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--loss", type=str, default="BCELoss", choices=["BCELoss", "DiceLoss", "JaccardLoss"],help="Learning rate")
    
    parser.add_argument("--l_max", type=int, default=12, help="Number of crooping loop, l_max in the paper")
    parser.add_argument("--dataset", type=str, default='d1', help="Dataset name")
    parser.add_argument("--data_root", type=str, default='./datasets', help="Dataset root path")
    parser.add_argument("--in_size", type=int, default=256, help="Train time input size")
    
    
    
    args = parser.parse_args()

    main(args)