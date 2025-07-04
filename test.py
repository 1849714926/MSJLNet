from __future__ import print_function
import argparse
import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from model import embed_net
from utils import *
import pdb
import scipy.io
import build_transforms

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline: resnet50')
parser.add_argument('--resume', '-r', default='sysu_base_p4_n4_lr_0.1_seed_0_plm_True_nums_10_lambda2_0.01_best.t', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')
parser.add_argument('--save_epoch', default=20, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str, help='log save path')
parser.add_argument('--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=192, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int, metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--margin', default=0.3, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=5, type=int, help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor for sysu')
parser.add_argument('--tvsearch', action='store_true', default = True, help='whether thermal to visible search on RegDB')
parser.add_argument('--gem_pool', default=True, type=bool)
parser.add_argument('--lambda_1', default=0.7, type=float, help='weights for tca')
parser.add_argument('--lambda_2', default=0.01, type=float, help='weights for loss_ch')
parser.add_argument('--ipm', default=True, type=bool)
parser.add_argument('--jslm', default=True, type=bool)
parser.add_argument('--part_nums', default=10, type=int)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

dataset = args.dataset
if dataset == 'sysu':
    data_path = 'D:\\Projects\\dataset\\SYSU-MM01\\'
    n_class = 395
    test_mode = [1, 2]
elif dataset =='regdb':
    data_path = './Datasets/RegDB/'
    n_class = 206
    test_mode = [2, 1]
elif dataset == 'llcm':
    data_path = './dataset/LLCM/'
    n_class = 713
    test_mode = [2, 1]  # [1, 2]: IR to VIS; [2, 1]: VIS to IR;

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0 
pool_dim = 2048
print('==> Building model..')
net = embed_net(n_class, datasets=args.dataset, arch=args.arch, ipm = args.ipm, jslm = args.jslm, pool_dim=pool_dim, part_num=args.part_nums)
net.to(device)    
cudnn.benchmark = True

checkpoint_path = args.model_path

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = build_transforms.test_transforms(
    args.img_h, args.img_w, normalize)
transform_color1 = build_transforms.train_transforms_color1(
    args.img_h, args.img_w, normalize)
transform_color2 = build_transforms.train_transforms_color2(
    args.img_h, args.img_w, normalize)
transform_thermal1 = build_transforms.train_transforms_thermal1(
    args.img_h, args.img_w, normalize)
transform_thermal2 = build_transforms.train_transforms_thermal2(
    args.img_h, args.img_w, normalize)
transform_train = transform_color1, transform_color2, transform_thermal1, transform_thermal2

end = time.time()

def extract_gall_feat(gall_loader):
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    k = (args.part_nums + 1) if args.plm else 1
    gall_feat1 = np.zeros((ngall, pool_dim))
    gall_feat2 = np.zeros((ngall, pool_dim * k))
    gall_feat3 = np.zeros((ngall, pool_dim))
    gall_feat4 = np.zeros((ngall, pool_dim * k))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            with torch.amp.autocast('cuda'):
                feat, feat_att = net(input, input, input, input, test_mode[0])
            gall_feat1[ptr:ptr + batch_num, :] = feat[:batch_num].detach().cpu().numpy()
            gall_feat2[ptr:ptr + batch_num, :] = feat_att[:batch_num].detach().cpu().numpy()
            gall_feat3[ptr:ptr + batch_num, :] = feat[batch_num:batch_num * 2].detach().cpu().numpy()
            gall_feat4[ptr:ptr + batch_num, :] = feat_att[batch_num:batch_num * 2].detach().cpu().numpy()
            # gall_feat5[ptr:ptr + batch_num, :] = feat[batch_num*2:].detach().cpu().numpy()
            # gall_feat6[ptr:ptr + batch_num, :] = feat_att[batch_num*2:].detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return gall_feat1, gall_feat2, gall_feat3, gall_feat4
    
def extract_query_feat(query_loader):
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    k = (args.part_nums + 1) if args.plm else 1
    query_feat1 = np.zeros((nquery, pool_dim))
    query_feat2 = np.zeros((nquery, pool_dim * k))
    query_feat3 = np.zeros((nquery, pool_dim))
    query_feat4 = np.zeros((nquery, pool_dim * k))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            with torch.amp.autocast('cuda'):
                feat, feat_att = net(input, input, input, input, test_mode[1])
            query_feat1[ptr:ptr + batch_num, :] = feat[:batch_num].detach().cpu().numpy()
            query_feat2[ptr:ptr + batch_num, :] = feat_att[:batch_num].detach().cpu().numpy()
            query_feat3[ptr:ptr + batch_num, :] = feat[batch_num:batch_num * 2].detach().cpu().numpy()
            query_feat4[ptr:ptr + batch_num, :] = feat_att[batch_num:batch_num * 2].detach().cpu().numpy()
            # query_feat5[ptr:ptr + batch_num, :] = feat[batch_num*2:].detach().cpu().numpy()
            # query_feat6[ptr:ptr + batch_num, :] = feat_att[batch_num*2:].detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return query_feat1, query_feat2, query_feat3, query_feat4


if dataset == 'sysu':

    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
        model_path = checkpoint_path + args.resume
        # model_path = checkpoint_path + 'sysu_awg_p4_n8_lr_0.1_seed_0_best.t'
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=0)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat1, query_feat2, query_feat_att1, query_feat_att2= extract_query_feat(query_loader)
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=trial)

        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=0)

        gall_feat1, gall_feat2, gall_feat_att1, gall_feat_att2 = extract_gall_feat(trial_gall_loader)

        # fc feature
        distmat1 = np.matmul(query_feat1, np.transpose(gall_feat1))
        distmat4 = np.matmul(query_feat2, np.transpose(gall_feat2))

        distmat_att1 = np.matmul(query_feat_att1, np.transpose(gall_feat_att1))
        distmat_att4 = np.matmul(query_feat_att2, np.transpose(gall_feat_att2))


        cmc_att1, mAP_att1, mINP_att1 = eval_sysu(-distmat_att1, query_label, gall_label, query_cam, gall_cam)
        cmc_att4, mAP_att4, mINP_att4 = eval_sysu(-distmat_att4, query_label, gall_label, query_cam, gall_cam)

        if trial == 0:
            all_cmc1, all_mAP1, all_mINP1 = cmc_att1, mAP_att1, mINP_att1
            all_cmc4, all_mAP4, all_mINP4 = cmc_att4, mAP_att4, mINP_att4

        else:
            all_cmc1 = all_cmc1 + cmc_att1
            all_mAP1 = all_mAP1 + mAP_att1
            all_mINP1 = all_mINP1 + mINP_att1


            all_cmc4 = all_cmc4 + cmc_att4
            all_mAP4 = all_mAP4 + mAP_att4
            all_mINP4 = all_mINP4 + mINP_att4


        print('Test Trial: {}'.format(trial))

        print(
            'VIS-IR FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att1[0], cmc_att1[4], cmc_att1[9], cmc_att1[19], mAP_att1, mINP_att1))
        print(
            'MIS-MR FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att4[0], cmc_att4[4], cmc_att4[9], cmc_att4[19], mAP_att4, mINP_att4))



elif dataset == 'regdb':

    for trial in range(10):
        test_trial = trial +1
        #model_path = checkpoint_path +  args.resume regdb_agw_p4_n4_lr_0.1_seed_0_trial_9_best.t
        model_path = checkpoint_path + 'regdb_base_p4_n6_lr_0.1_seed_0_trial_{}_best.t'.format(test_trial)
        print(model_path)
        print(os.path.isfile(model_path))
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])
        else:
            print('==> no checkpoint found at {}'.format(model_path))


        # training set
        trainset = RegDBData(data_path, test_trial, transform=transform_train)
        # generate the idx of each person identity
        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

        # testing set
        query_img, query_label = process_test_regdb(data_path, trial=test_trial, modal='visible')
        gall_img, gall_label = process_test_regdb(data_path, trial=test_trial, modal='thermal')

        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        nquery = len(query_label)
        ngall = len(gall_label)

        queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
        print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

        gall_feat1, gall_feat2, gall_feat_att1, gall_feat_att2 = extract_gall_feat(gall_loader)
        query_feat1, query_feat2, query_feat_att1, query_feat_att2 = extract_query_feat(query_loader)


        if args.tvsearch:
            distmat1 = np.matmul(query_feat1, np.transpose(gall_feat1))
            distmat2 = np.matmul(query_feat1, np.transpose(gall_feat2))
            distmat3 = np.matmul(query_feat2, np.transpose(gall_feat1))
            distmat4 = np.matmul(query_feat2, np.transpose(gall_feat2))

            distmat_att1 = np.matmul(query_feat_att1, np.transpose(gall_feat_att1))
            distmat_att2 = np.matmul(query_feat_att1, np.transpose(gall_feat_att2))
            distmat_att3 = np.matmul(query_feat_att2, np.transpose(gall_feat_att1))
            distmat_att4 = np.matmul(query_feat_att2, np.transpose(gall_feat_att2))

            distmat_total = distmat1 + distmat2 + distmat3 + distmat4
            distmat_total_att = distmat_att1 + distmat_att2 + distmat_att3 + distmat_att4
            distmat_all = distmat1 + distmat_att1 + distmat2 + distmat_att2 + distmat3 + distmat_att3 + distmat4 + distmat_att4

            cmc_att1, mAP_att1, mINP_att1 = eval_regdb(-distmat_att1, gall_label, query_label)
            cmc_att2, mAP_att2, mINP_att2 = eval_regdb(-distmat_att2, gall_label, query_label)
            cmc_att3, mAP_att3, mINP_att3 = eval_regdb(-distmat_att3, gall_label, query_label)
            cmc_att4, mAP_att4, mINP_att4 = eval_regdb(-distmat_att4, gall_label, query_label)
            cmc_att5, mAP_att5, mINP_att5 = eval_regdb(-distmat_total_att, gall_label, query_label)
            cmc, mAP, mINP = eval_regdb(-distmat_all, gall_label, query_label)

        else:
            distmat1 = np.matmul(gall_feat1, np.transpose(query_feat1))
            distmat2 = np.matmul(gall_feat1, np.transpose(query_feat1))
            distmat3 = np.matmul(gall_feat1, np.transpose(query_feat1))
            distmat4 = np.matmul(gall_feat1, np.transpose(query_feat1))

            distmat_att1 = np.matmul(gall_feat_att1, np.transpose(query_feat_att1))
            distmat_att2 = np.matmul(gall_feat_att1, np.transpose(query_feat_att2))
            distmat_att3 = np.matmul(gall_feat_att2, np.transpose(query_feat_att1))
            distmat_att4 = np.matmul(gall_feat_att2, np.transpose(query_feat_att2))

            distmat_total = distmat1 + distmat2 + distmat3 + distmat4
            distmat_total_att = distmat_att1 + distmat_att2 + distmat_att3 + distmat_att4
            distmat_all = distmat1 + distmat_att1 + distmat2 + distmat_att2 + distmat3 + distmat_att3 + distmat4 + distmat_att4

            cmc_att1, mAP_att1, mINP_att1 = eval_regdb(-distmat_att1, query_label, gall_label)
            cmc_att2, mAP_att2, mINP_att2 = eval_regdb(-distmat_att2, query_label, gall_label)
            cmc_att3, mAP_att3, mINP_att3 = eval_regdb(-distmat_att3, query_label, gall_label)
            cmc_att4, mAP_att4, mINP_att4 = eval_regdb(-distmat_att4, query_label, gall_label)
            cmc_att5, mAP_att5, mINP_att5 = eval_regdb(-distmat_total_att, query_label, gall_label)
            cmc, mAP, mINP = eval_regdb(-distmat_all, query_label, gall_label)

        if trial == 0:
            all_cmc1, all_mAP1, all_mINP1 = cmc_att1, mAP_att1, mINP_att1
            all_cmc2, all_mAP2, all_mINP2 = cmc_att2, mAP_att2, mINP_att2
            all_cmc3, all_mAP3, all_mINP3 = cmc_att3, mAP_att3, mINP_att3
            all_cmc4, all_mAP4, all_mINP4 = cmc_att4, mAP_att4, mINP_att4
            all_cmc5, all_mAP5, all_mINP5 = cmc_att5, mAP_att5, mINP_att5
            all_cmc, all_mAP, all_mINP = cmc, mAP, mINP

        else:
            all_cmc1 = all_cmc1 + cmc_att1
            all_mAP1 = all_mAP1 + mAP_att1
            all_mINP1 = all_mINP1 + mINP_att1

            all_cmc2 = all_cmc2 + cmc_att2
            all_mAP2 = all_mAP2 + mAP_att2
            all_mINP2 = all_mINP2 + mINP_att2

            all_cmc3 = all_cmc3 + cmc_att3
            all_mAP3 = all_mAP3 + mAP_att3
            all_mINP3 = all_mINP3 + mINP_att3

            all_cmc4 = all_cmc4 + cmc_att4
            all_mAP4 = all_mAP4 + mAP_att4
            all_mINP4 = all_mINP4 + mINP_att4

            all_cmc5 = all_cmc5 + cmc_att5
            all_mAP5 = all_mAP5 + mAP_att5
            all_mINP5 = all_mINP5 + mINP_att5

            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP

        print('Test Trial: {}'.format(trial))

        print(
            'VIS-IR FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att1[0], cmc_att1[4], cmc_att1[9], cmc_att1[19], mAP_att1, mINP_att1))
        print(
            'VIS-MR FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att2[0], cmc_att2[4], cmc_att2[9], cmc_att2[19], mAP_att2, mINP_att2))
        print(
            'MIS-IR FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att3[0], cmc_att3[4], cmc_att3[9], cmc_att3[19], mAP_att3, mINP_att3))
        print(
            'MIS-MR FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att4[0], cmc_att4[4], cmc_att4[9], cmc_att4[19], mAP_att4, mINP_att4))
        print(
            'ALL FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att5[0], cmc_att5[4], cmc_att5[9], cmc_att5[19], mAP_att5, mINP_att5))
        print(
            'ALL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

elif dataset == 'llcm':
    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
        model_path = checkpoint_path + args.resume
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # testing set
    query_img, query_label, query_cam = process_query_llcm(data_path, mode=test_mode[1])
    gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=0)

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat1, query_feat2, query_feat_att1, query_feat_att2= extract_query_feat(query_loader)
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=trial)

        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat1, gall_feat2, gall_feat_att1, gall_feat_att2 = extract_gall_feat(trial_gall_loader)

        # fc feature
        if test_mode == [1, 2]:
            distmat1 = np.matmul(query_feat1, np.transpose(gall_feat1))
            distmat2 = np.matmul(query_feat1, np.transpose(gall_feat2))
            distmat3 = np.matmul(query_feat2, np.transpose(gall_feat1))
            distmat4 = np.matmul(query_feat2, np.transpose(gall_feat2))

            distmat_att1 = np.matmul(query_feat_att1, np.transpose(gall_feat_att1))
            distmat_att2 = np.matmul(query_feat_att1, np.transpose(gall_feat_att2))
            distmat_att3 = np.matmul(query_feat_att2, np.transpose(gall_feat_att1))
            distmat_att4 = np.matmul(query_feat_att2, np.transpose(gall_feat_att2))

            distmat_total = distmat1 + distmat2 + distmat3 + distmat4
            distmat_total_att = distmat_att1 + distmat_att2 + distmat_att3 + distmat_att4
            distmat_all = distmat1 + distmat_att1 + distmat2 + distmat_att2 + distmat3 + distmat_att3 + distmat4 + distmat_att4

            cmc_att1, mAP_att1, mINP_att1 = eval_llcm(-distmat_att1, query_label, gall_label, query_cam, gall_cam)
            cmc_att2, mAP_att2, mINP_att2 = eval_llcm(-distmat_att2, query_label, gall_label, query_cam, gall_cam)
            cmc_att3, mAP_att3, mINP_att3 = eval_llcm(-distmat_att3, query_label, gall_label, query_cam, gall_cam)
            cmc_att4, mAP_att4, mINP_att4 = eval_llcm(-distmat_att4, query_label, gall_label, query_cam, gall_cam)
            cmc_att5, mAP_att5, mINP_att5 = eval_llcm(-distmat_total_att, query_label, gall_label, query_cam, gall_cam)
            cmc, mAP, mINP = eval_llcm(-distmat_all, query_label, gall_label, query_cam, gall_cam)
        else:
            distmat1 = np.matmul(gall_feat1, np.transpose(query_feat1))
            distmat2 = np.matmul(gall_feat1, np.transpose(query_feat1))
            distmat3 = np.matmul(gall_feat1, np.transpose(query_feat1))
            distmat4 = np.matmul(gall_feat1, np.transpose(query_feat1))

            distmat_att1 = np.matmul(gall_feat_att1, np.transpose(query_feat_att1))
            distmat_att2 = np.matmul(gall_feat_att1, np.transpose(query_feat_att2))
            distmat_att3 = np.matmul(gall_feat_att2, np.transpose(query_feat_att1))
            distmat_att4 = np.matmul(gall_feat_att2, np.transpose(query_feat_att2))

            distmat_total = distmat1 + distmat2 + distmat3 + distmat4
            distmat_total_att = distmat_att1 + distmat_att2 + distmat_att3 + distmat_att4
            distmat_all = distmat1 + distmat_att1 + distmat2 + distmat_att2 + distmat3 + distmat_att3 + distmat4 + distmat_att4

            cmc_att1, mAP_att1, mINP_att1 = eval_llcm(-distmat_att1, gall_label, query_label, gall_cam, query_cam)
            cmc_att2, mAP_att2, mINP_att2 = eval_llcm(-distmat_att2, gall_label, query_label, gall_cam, query_cam)
            cmc_att3, mAP_att3, mINP_att3 = eval_llcm(-distmat_att3, gall_label, query_label, gall_cam, query_cam)
            cmc_att4, mAP_att4, mINP_att4 = eval_llcm(-distmat_att4, gall_label, query_label, gall_cam, query_cam)
            cmc_att5, mAP_att5, mINP_att5 = eval_llcm(-distmat_total_att, gall_label, query_label, gall_cam, query_cam)
            cmc, mAP, mINP = eval_llcm(-distmat_all, gall_label, query_label, gall_cam, query_cam)

        if trial == 0:
            all_cmc1, all_mAP1, all_mINP1 = cmc_att1, mAP_att1, mINP_att1
            all_cmc2, all_mAP2, all_mINP2 = cmc_att2, mAP_att2, mINP_att2
            all_cmc3, all_mAP3, all_mINP3 = cmc_att3, mAP_att3, mINP_att3
            all_cmc4, all_mAP4, all_mINP4 = cmc_att4, mAP_att4, mINP_att4
            all_cmc5, all_mAP5, all_mINP5 = cmc_att5, mAP_att5, mINP_att5
            all_cmc, all_mAP, all_mINP = cmc, mAP, mINP

        else:
            all_cmc1 = all_cmc1 + cmc_att1
            all_mAP1 = all_mAP1 + mAP_att1
            all_mINP1 = all_mINP1 + mINP_att1

            all_cmc2 = all_cmc2 + cmc_att2
            all_mAP2 = all_mAP2 + mAP_att2
            all_mINP2 = all_mINP2 + mINP_att2

            all_cmc3 = all_cmc3 + cmc_att3
            all_mAP3 = all_mAP3 + mAP_att3
            all_mINP3 = all_mINP3 + mINP_att3

            all_cmc4 = all_cmc4 + cmc_att4
            all_mAP4 = all_mAP4 + mAP_att4
            all_mINP4 = all_mINP4 + mINP_att4

            all_cmc5 = all_cmc5 + cmc_att5
            all_mAP5 = all_mAP5 + mAP_att5
            all_mINP5 = all_mINP5 + mINP_att5

            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP

        print('Test Trial: {}'.format(trial))

        print(
            'VIS-IR FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att1[0], cmc_att1[4], cmc_att1[9], cmc_att1[19], mAP_att1, mINP_att1))
        print(
            'VIS-MR FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att2[0], cmc_att2[4], cmc_att2[9], cmc_att2[19], mAP_att2, mINP_att2))
        print(
            'MIS-IR FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att3[0], cmc_att3[4], cmc_att3[9], cmc_att3[19], mAP_att3, mINP_att3))
        print(
            'MIS-MR FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att4[0], cmc_att4[4], cmc_att4[9], cmc_att4[19], mAP_att4, mINP_att4))
        print(
            'ALL FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att5[0], cmc_att5[4], cmc_att5[9], cmc_att5[19], mAP_att5, mINP_att5))
        print(
            'ALL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

cmc1, mAP1, mINP1 = all_cmc1 / 10, all_mAP1 / 10, all_mINP1 / 10
cmc4, mAP4, mINP4 = all_cmc4 / 10, all_mAP4 / 10, all_mINP4 / 10

print('All Average:')

print(
    'VIS-IR FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc1[0], cmc1[4], cmc1[9], cmc1[19], mAP1, mINP1))
print(
    'MIS-MR FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc4[0], cmc4[4], cmc4[9], cmc4[19], mAP4, mINP4))