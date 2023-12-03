import torch
import logging
import sys
import os
# import torchvision.models as models
import torch.backends.cudnn as cudnn
import numpy as np
import torch.optim as optim
import argparse
import shutil
import pickle
import random
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
import torch.utils.data.sampler as sampler
from tqdm import tqdm
from scipy.ndimage import zoom
from utils import losses, metrics, ramps
from model_3d_UneTr import create_model_3d
from dataloaders.dataset_3d import *
from torchvision.utils import make_grid

parser = argparse.ArgumentParser("")
parser.add_argument('--root_path', type=str,
                    default='/home/chenyu/Documents/UA-MT/data', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='LA/fully_supervised', help='experiment_name')
parser.add_argument('--resume', type=str,
                    default='', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='vnet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
# parser.add_argument('--patch_size', type=list,  default=[256, 256],
#                     help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
# parser.add_argument('--num_classes', type=int,  default=4,
#                     help='output channel of network')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--fc_dim', type=int, default=256)
parser.add_argument('--num_fc', type=int, default=3)
parser.add_argument('--input_channels', type=int, default=256)
parser.add_argument('--coarse_pred_each_layer', type=eval, default=True, choices=[True, False])
parser.add_argument('--cls_agnostic_mask', type=eval, default=False, choices=[True, False])

parser.add_argument('--train_num_points', type=int, default=2048)
parser.add_argument('--in_features', type=int, default=256)
parser.add_argument('--oversample_ratio', type=float, default=3)
parser.add_argument('--importance_sample_ratio', type=float, default=0.75)
parser.add_argument('--subdivision_steps', type=int, default=2)
parser.add_argument('--subdivision_num_points', type=int, default=8192)

args = parser.parse_args()
patch_size = (112, 112, 80)
num_classes = 2

def train(args, snapshot_path):
    base_lr = args.base_lr
    # num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    
    db_train = LAHeart(base_dir=args.root_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    
        
    total_slices = len(db_train)
    print("Total silices is: {}".format(total_slices))
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    
    model = create_model_3d(num_classes=num_classes).cuda()
    # model.model.load_state_dict(torch.load(args.resume))
    
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            
            # loss = model(volume_batch, label_batch)
            
            outputs = model(volume_batch)[0]
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss = loss_ce + loss_dice
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
                
            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            
            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            
            if iter_num % 20 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)
                # outputs = torch.argmax(torch.softmax(
                #     outputs, dim=1), dim=1, keepdim=True)
                # model.eval()
                # res = model(volume_batch, label_batch)
                # outputs = torch.argmax(torch.softmax(
                #     res, dim=1), dim=1, keepdim=True)
                # writer.add_image('train/Prediction',
                #                  outputs[1, ...] * 50, iter_num)
                # model.train()
                
            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"



if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, 100, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)

        
        
