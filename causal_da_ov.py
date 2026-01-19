import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network
import loss 
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx,ImageList_idx_aug_fix
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import clip
import torch.nn.functional as F
import pandas as pd
import IID_losses as iid_loss
import miro
from PIL import Image
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight
import time
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from data.imagnet_prompts import imagenet_classes
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
from clip.custom_clip import get_coop
from copy import deepcopy
from sklearn import metrics




def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(args,optimizer, iter_num, max_iter, gamma=10, power=0.75):
    if (args.dset =='office-home'):
        # print(1)
        decay = (1 + gamma * iter_num / max_iter) ** (-power)
    else :
        decay = 1
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
#   else:
#     normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
#   else:
#     normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])
def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'   
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'   
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx_aug_fix(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    dsets["test"] = ImageList_idx_aug_fix(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0][0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent

def train_target(args):
    text_inputs = clip_pre_text(args.FILE)
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()  

    model = get_coop(args.arch, args.dset, args.gpu, args.n_ctx, args.ctx_init)
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'    
    netC.load_state_dict(torch.load(modelpath))
    target_logits = torch.ones(args.batch_size,args.class_num)
    im_re_o = miro.MIRO(target_logits.shape).cuda()
    im_ib_o = miro.MIRO(target_logits.shape).cuda()
    del target_logits

    for name, param in model.named_parameters():
        if not args.cocoop:
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        else:
            if "text_encoder" not in name:
                param.requires_grad_(False)

    param_group = []
    param_group_ib = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False

    for k, v in netC.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False

    for k, v in im_ib_o.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False


    for k, v in im_re_o.named_parameters():
        if(v.requires_grad == True):
            param_group_ib += [{'params': v, 'lr': args.lr * args.lr_decay3}]

    for k, v in model.prompt_learner.named_parameters():
        if(v.requires_grad == True):
            param_group_ib += [{'params': v, 'lr': args.lr * args.lr_decay3}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    optimizer_ib = optim.SGD(param_group_ib)
    optimizer_ib = op_copy(optimizer_ib)
    optim_state = deepcopy(optimizer_ib.state_dict())


    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0
    classnames = args.classname
    model.reset_classnames(classnames, args.arch)
    start = True
    epoch = 0 
    
    while iter_num < max_iter:
        try:
            (inputs_test,clip_inputs_test),labels, tar_idx = next(iter_test)
        except:
            iter_test = iter(dset_loaders["target"])
            (inputs_test,clip_inputs_test),labels, tar_idx = next(iter_test)

        if inputs_test.size(0) == 1:
            continue
        
        inputs_test = inputs_test.cuda()
        clip_inputs_test = clip_inputs_test[0].cuda()

        iter_num += 1
        lr_scheduler(args,optimizer, iter_num=iter_num, max_iter=max_iter)
        with torch.no_grad():
            outputs_test_new = netC(netB(netF(inputs_test))).detach()
        netF.eval()
        netB.eval()
        netC.eval()
        model.train()
        im_re_o.train()
        
        output_clip,_ = test_time_adapt_eval(clip_inputs_test,labels, model, optimizer_ib, optim_state,args,outputs_test_new,im_re_o)
        output_clip = output_clip.detach().cuda().float()
        output_clip_sm = nn.Softmax(dim=1)(output_clip)

        netF.train()
        netB.train()
        netC.train()
        model.eval()
        im_re_o.eval()
        outputs_test = netC(netB(netF(inputs_test)))
        softmax_out = nn.Softmax(dim=1)(outputs_test)

        
        if (args.loss_func=="l1"):
            loss_l1 = torch.nn.L1Loss(reduction='mean')
            classifier_loss = loss_l1(softmax_out, output_clip_sm)
            classifier_loss *= args.cls_par
        elif (args.loss_func=="l2"):
            loss_l2 = torch.nn.MSELoss(reduction='mean')
            classifier_loss = loss_l2(softmax_out,output_clip_sm)
            classifier_loss *= args.cls_par
        elif (args.loss_func=="iid"):
            classifier_loss = iid_loss.IID_loss(softmax_out,output_clip_sm)
            classifier_loss *= args.cls_par
        elif (args.loss_func=="kl"):
            classifier_loss = F.kl_div(softmax_out.log(),output_clip_sm, reduction='sum')
            classifier_loss *= args.cls_par
        elif (args.loss_func=="sce"):
            _, pred = torch.max(output_clip, 1)
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= args.cls_par
        else :
            classifier_loss = torch.tensor(0.0).cuda()


        if args.ent:
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                if args.dset == 'VISDA-C' :
                    msoftmax = softmax_out.mean(dim=0)
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                    entropy_loss -= 0.5*gentropy_loss
                else:
                    msoftmax = softmax_out.mean(dim=0)
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                    entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()
        with torch.no_grad():
            if start:
                all_output_clip = output_clip.float().cpu()
                all_label = labels.float()
                start = False
            else:
                all_output_clip = torch.cat((all_output_clip, output_clip.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            start = True
            epoch = epoch + 1
            _, clip_predict = torch.max(all_output_clip, 1)
            accuracy = torch.sum(torch.squeeze(clip_predict).float() == all_label).item() / float(all_label.size()[0])
            accuracy = accuracy*100
            print(accuracy)
            log_str ='CLIP_Accuracy = {:.2f}%'.format(accuracy)
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            netF.eval()
            netB.eval()
            netC.eval()
            im_re_o.eval()
            model.eval()

            if args.dset=='VISDA-C':
                acc_s_te, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te) + '\n' + acc_list
            else:
                acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
                log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str+'\n')
            netF.train()
            netB.train()
            netC.train()


    if args.issave:   
        torch.save(netF.state_dict(), osp.join(args.output_dir, "target_F_" + args.savename + ".pt"))
        torch.save(netB.state_dict(), osp.join(args.output_dir, "target_B_" + args.savename + ".pt"))
        torch.save(netC.state_dict(), osp.join(args.output_dir, "target_C_" + args.savename + ".pt"))
        
    return netF, netB, netC

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def test_time_tuning(model, inputs, optimizer, args,target_output,im_re_o):

    target_output = target_output.cuda()

    for j in range(args.tta_steps):
        with torch.cuda.amp.autocast():

            output_logits,_,text_features = model(inputs)             
            if(output_logits.shape[0]!=args.batch_size):
                padding_f=torch.zeros([args.batch_size-output_logits.shape[0],output_logits.shape[1]],dtype=torch.float).cuda()
                output_logits = torch.cat((output_logits, padding_f.float()), 0)
                target_output =  torch.cat((target_output, padding_f.float()), 0)

            im_loss_o, Delta = im_re_o.update(output_logits,target_output)
            Delta = 1.0/(Delta+1e-5)
            Delta = nn.Softmax(dim=1)(Delta)
            output_logits_sm = nn.Softmax(dim=1)(output_logits)
            output = Delta*output_logits_sm
            iic_loss = iid_loss.IID_loss(output, output_logits_sm)
            loss = 0.5*(iic_loss - 0.0003*im_loss_o) #0.0003

            if(inputs.shape[0]!=args.batch_size):
                output = output[:inputs.shape[0]]
                target_output = target_output[:inputs.shape[0]]



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    return output,loss


def test_time_adapt_eval(input,target, model, optimizer, optim_state, args,target_output,im_re_o):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    start_test = True
    progress = ProgressMeter(
        input.shape[0],
        [batch_time, top1, top5],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()
    if not args.cocoop: # no need to reset cocoop because it's fixed
        with torch.no_grad():
            model.train()
            im_re_o.train()
    end = time.time()

        
    if not args.cocoop: # no need to reset cocoop because it's fixed
        if args.tta_steps > 0:
            with torch.no_grad():
                model.train()
                im_re_o.train()
        optimizer.load_state_dict(optim_state)
        output,loss_ib = test_time_tuning(model, input, optimizer, args,target_output,im_re_o)

    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            model.eval()
            im_re_o.eval()
            output,_,_ = model(input)
    output = output.cpu()

    batch_time.update(time.time() - end)
    end = time.time()

    return output,loss_ib

def clip_pre(text_inputs,inputs_test):
    with torch.no_grad():
        image_features = clip_model.encode_image(inputs_test)
        logits_per_image, logits_per_text = clip_model(inputs_test, text_inputs)
        # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        probs = logits_per_image.cpu()
    return image_features,probs

def clip_text(text_features,inputs_test):
    with torch.no_grad():
        image_features = clip_model.encode_image(inputs_test)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T)
    return image_features,similarity


def clip_pre_text(FILE):
    List_rd = []
    with open(FILE) as f:
        for line in f:
            List_rd.extend([i for i in line.split()])
    f.close()
    classnames = List_rd
    classnames = [name.replace("_", " ") for name in classnames]
    args.classname = classnames
    prompt_prefix = args.ctx_init.replace("_"," ")
    prompts = [prompt_prefix + " " + name + "." for name in classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
    return tokenized_prompts

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['VISDA-C', 'office', 'office-home', 'office-caltech','Domain-Net'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--loss_func', type=str, default='sce')
    parser.add_argument('--gent', action='store_false')
    parser.add_argument('--ent', action='store_false')
    parser.add_argument('--tent_ent', action='store_true')
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.4)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--lr_decay3', type=float, default=0.01)
    parser.add_argument('--bottleneck', type=int, default=512)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='BYX/test_mean')
    parser.add_argument('--output_src', type=str, default='/home/imi/data1/project/Modelzoom/source')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    # parser.add_argument('--issave', action='store_true')
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('--arch', type=str, default='ViT-B/32', choices=['RN50', 'ViT-B/32','RN101','ViT-B/16'])
    parser.add_argument('--FILE', type=str, default='./data/office/amazon_list_code2.txt')
    
    parser.add_argument('--data', default='/home/ts/projects/Datazoom/',metavar='--DIR', help='path to dataset root')
    # parser.add_argument('--test_sets', type=str, default='office', help='test dataset (multiple datasets split by slash)') #改
    # parser.add_argument('--domain_num', type=int, default=0, help='domain_num')#改
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    # parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/32')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('--lr_TIP', default=0.01, type=float,
                         help='initial learning rate')
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--classnames', type=str, default='./data/office/amazon_list_code2.txt', help='classnames')
    parser.add_argument('--tpt', action='store_true', default=True, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default='a_photo_of_a', type=str, help='init tunable prompts')
    parser.add_argument('--cocoop', action='store_true', default=False, help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--load_2', default=None, type=str, help='path to a pre-trained coop/cocoop')

    parser.add_argument('--output_dir', type=str, default='./result/')
    parser.add_argument('--confi', type=bool, default=True)
    parser.add_argument('--init_last', type=bool, default=False)
    parser.add_argument('--confi_rate', type=float, default=0.2)
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.FILE = './data/office-home/RealWorld_list_code2.txt'
        args.class_num = 65 
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.FILE = './data/office/amazon_list_code2.txt'
        args.class_num = 31
    if args.dset == 'VISDA-C':
        names = ['train', 'validation']
        args.FILE = './data/VISDA-C/validation_list_code.txt'
        args.class_num = 12
    if args.dset == 'office-caltech':
        names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    if args.dset == 'Domain-Net':
        names = ['clipart', 'infograph', 'painting', 'quickdraw','real','sketch']
        args.FILE = './data/Domain-Net/c_list.txt_code.txt'
        args.class_num = 345
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    
    print("+++++++++++++++++++SHOT+++++++++++++++++++++")
    args.type = names
    folder = './data/'
    args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
    args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
    args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

    args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
    args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
    args.name = names[args.s][0].upper()+names[args.t][0].upper()

    if args.dset == 'office-home':
        if args.da == 'pda':
            args.class_num = 65
            args.src_classes = [i for i in range(65)]
            args.tar_classes = [i for i in range(25)]

    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.savename = 'par_' + str(args.cls_par)
    if args.da == 'pda':
        args.gent = ''
        args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
    args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
    args.out_file.write(print_args(args)+'\n')
    args.out_file.flush()
    DOWNLOAD_ROOT = '~/.cache/clip'
    device = int(args.gpu_id)
    clip_model, preprocess,_ = clip.load(args.arch, device=device, download_root=DOWNLOAD_ROOT)
    clip_model.float()
    clip_model.eval()
    train_target(args)