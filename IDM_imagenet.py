import os
import random
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from dc_utils import parser_bool, downscale, epoch_no_loader, evaluate_synset_edc, get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug, number_sign_augment, padding_augment
import torchnet
import torch.nn.functional as F

def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DM', help='DC/DSA/DM')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ResNet10_AP', help='model')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=20000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='/data1/zhaoganlong/dataset/imagenet', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--outer_loop', type=int, default=1, help='outer loop for network update')
    parser.add_argument('--inner_loop', type=int, default=1, help='outer loop for network update')
    parser.add_argument('--eval_interval', type=int, default=100, help='outer loop for network update')
    parser_bool(parser, 'net_train_real', False)
    parser.add_argument('--load_path', type=str, default='None', help='distance metric')
    parser.add_argument('--trained_bs', type=int, default=256, help='outer loop for network update')
    parser.add_argument('--last', type=int, default=100, help='outer loop for network update')
    parser.add_argument('--mismatch_lambda', type=float, default=0.005, help='outer loop for network update')
    parser_bool(parser, 'net_decay', False)
    parser.add_argument('--model_train_steps', type=int, default=10, help='outer loop for network update')
    parser.add_argument('--net_num', type=int, default=100, help='outer loop for network update')
    parser.add_argument('--net_begin', type=int, default=0, help='outer loop for network update')
    parser.add_argument('--net_end', type=int, default=100000, help='outer loop for network update')
    parser.add_argument('--mismatch_type', type=str, default='l1', help='outer loop for network update')
    parser.add_argument('--ij_selection', type=str, default='random', help='outer loop for network update')
    parser.add_argument('--weight_layer_index', type=int, default=0, help='outer loop for network update')
    parser.add_argument('--net_generate_interval', type=int, default=30, help='outer loop for network update')
    parser.add_argument('--fetch_net_num', type=int, default=2, help='outer loop for network update')
    parser.add_argument('--embed_last', type=int, default=-1, help='outer loop for network update')
    parser.add_argument('--perturbation_type', type=str, default='weight', help='outer loop for network update')
    parser.add_argument('--conf_path', type=str, default='', help='outer loop for network update')
    parser_bool(parser, 'local_match', False)
    parser.add_argument('--local_num', type=int, default=10, help='outer loop for network update')
    parser.add_argument('--local_weight', type=float, default=0.1, help='outer loop for network update')
    parser_bool(parser, 'center_reg', False)
    parser.add_argument('--center_reg_weight', type=float, default=0.01, help='outer loop for network update')
    parser_bool(parser, 'syn_ce', False)
    parser.add_argument('--ce_weight', type=float, default=0.1, help='outer loop for network update')
    parser.add_argument('--optim', type=str, default='sgd', help='outer loop for network update')
    parser.add_argument('--train_net_num', type=int, default=2, help='outer loop for network update')
    parser.add_argument('--aug_num', type=int, default=1, help='outer loop for network update')
    parser_bool(parser, 'aug', False)
    parser.add_argument('--eval_begin', type=int, default=1, help='outer loop for network update')
    parser_bool(parser, 'model_evaluate', False)

    args = parser.parse_args()
    # args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    eval_it_pool = np.arange(0, args.Iteration+1, args.eval_interval).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)
    model_eval_pool = get_eval_pool(args.eval_mode, 'ConvNet_6', args.model)

    print("Start Loading ImageNet Subset")
    from others_edc import load_resized_data, ClassDataLoader, ClassMemDataLoader
    args.load_memory = True
    args.size = 224
    args.nclass = 100
    args.phase = 4
    args.dseed = 1
    args.batch_size = 256
    args.workers = 8
    dst_train, testloader = load_resized_data(args)
    if args.load_memory:
        loader_real = ClassMemDataLoader(dst_train, batch_size=args.batch_real)
    else:
        loader_real = ClassDataLoader(dst_train,
                                      batch_size=args.batch_real,
                                      num_workers=args.workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)
    num_classes = dst_train.nclass
    channel, hs, ws = dst_train[0][0].shape
    im_size = (hs, ws)


    iter_loader_train = iter(loader_real)

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            if args.aug:
                for c in range(num_classes):
                    half_size = im_size[0]//2
                    image_syn.data[c*args.ipc:(c+1)*args.ipc, :, :half_size, :half_size] = downscale(loader_real.class_sample(c, args.ipc)[0], 0.5).detach().data
                    image_syn.data[c*args.ipc:(c+1)*args.ipc, :, half_size:, :half_size] = downscale(loader_real.class_sample(c, args.ipc)[0], 0.5).detach().data
                    image_syn.data[c*args.ipc:(c+1)*args.ipc, :, :half_size, half_size:] = downscale(loader_real.class_sample(c, args.ipc)[0], 0.5).detach().data
                    image_syn.data[c*args.ipc:(c+1)*args.ipc, :, half_size:, half_size:] = downscale(loader_real.class_sample(c, args.ipc)[0], 0.5).detach().data
            else:
                from tqdm import tqdm
                for c in tqdm(range(num_classes)):
                    # image_syn.data[c*args.ipc:(c+1)*args.ipc, :, :, :] = classwise_iter[c].next()[0][:args.ipc].detach().data
                    image_syn.data[c*args.ipc:(c+1)*args.ipc, :, :, :] = loader_real.class_sample(c, args.ipc)[0].detach().data
        else:
            raise NotImplemented()

        ''' training '''
        if args.optim == 'sgd':
            optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
        elif args.optim == 'adam':
            optimizer_img = torch.optim.Adam([image_syn, ], lr=args.lr_img)
        else:
            raise NotImplemented()
        optimizer_img.zero_grad()
        print('%s training begins'%get_time())

        ''' Train synthetic data '''
        net_num = args.net_num
        net_list = list()
        optimizer_list = list()
        acc_meters = list()
        for net_index in range(3):
            net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
            net.train()
            if args.net_decay:
                optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net, momentum=0.9, weight_decay=0.0005)
            else:
                optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            net_list.append(net)
            optimizer_list.append(optimizer_net)
            acc_meters.append(torchnet.meter.ClassErrorMeter(accuracy=True))
        
        criterion = nn.CrossEntropyLoss().to(args.device)

        for it in range(args.Iteration+1):

            ''' Evaluate synthetic data '''
            if it in eval_it_pool[args.eval_begin:]:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))

                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)

                    accs = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                        if args.aug:
                            image_syn_eval, label_syn_eval = number_sign_augment(image_syn_eval, label_syn_eval)
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, tqdm_bar=True)
                        accs.append(acc_test)
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs

                # if args.load_path is None or args.load_path == 'none' or args.load_path == 'None':
                if args.model_evaluate:
                    # reset accuracy meters
                    acc_report = "Training Acc: "
                    for acc_index, acc_meter in enumerate(acc_meters):
                        acc_report += 'Net {}: {:.2f}% '.format(acc_index, acc_meter.value()[0] if acc_meter.n != 0 else 0)
                        acc_meter.reset()
                    acc_report += ' Testing Acc: '
                    for net_index, test_model in enumerate(net_list):
                        loss_test, acc_test = epoch('test', testloader, test_model, None, criterion, args, aug = False)
                        acc_report += 'Net {}: {:.2f}% '.format(net_index, acc_test*100)
                    print(acc_report)


            if it % args.net_generate_interval == 0:
                # append and pop net list:
                if len(net_list) == net_num:
                    net_list.pop(0)
                    optimizer_list.pop(0)
                    acc_meters.pop(0)
                net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
                net.train()
                if args.net_decay:
                    optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net, momentum=0.9, weight_decay=0.0005)
                else:
                    optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
                optimizer_net.zero_grad()
                net_list.append(net)
                optimizer_list.append(optimizer_net)
                acc_meters.append(torchnet.meter.ClassErrorMeter(accuracy=True))

            _ = list(range(len(net_list)))
            if len(_[args.net_begin: args.net_end]) > 10:
                _ = _[args.net_begin: args.net_end]
            random.shuffle(_)
            if args.ij_selection == 'random':
                # net_index_i, net_index_j = _[:2]
                net_index_list = _[:args.train_net_num]
            elif args.ij_selection == 'adjoining':
                raise NotImplemented()
                net_index_i = _[0]
                net_index_j = (_[0] + 1) % net_num
            else:
                raise NotImplemented()
            train_net_list = [net_list[ind] for ind in net_index_list]
            train_acc_list = [acc_meters[ind] for ind in net_index_list]

            embed_list = [net.module.embed_channel_avg if torch.cuda.device_count() > 1 else net.embed_channel_avg for net in train_net_list]
            
            for _ in range(args.outer_loop):
                loss_avg = 0
                loss = torch.tensor(0.0).to(args.device)
                mtt_loss_avg = 0
                metrics = {'mtt': 0, 'syn': 0}
                acc_avg = {'mtt':torchnet.meter.ClassErrorMeter(accuracy=True), 'syn':torchnet.meter.ClassErrorMeter(accuracy=True)}

                ''' update synthetic data '''
                if 'BN' not in args.model or args.model == 'ConvNet_GBN': # for ConvNet
                    image_temp = image_syn
                    image_sign = 'syn'
                    for net_ind in range(len(train_net_list)):
                        net = train_net_list[net_ind]
                        net.eval()
                        embed = embed_list[net_ind]
                        net_acc = train_acc_list[net_ind]
                        for c in range(num_classes):
                            loss_c = torch.tensor(0.0).to(args.device)
                            img_real, _ = loader_real.class_sample(c, args.batch_real)
                            assert (_ == c).all()
                            img_syn = image_temp[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                            lab_syn = label_syn[c*args.ipc:(c+1)*args.ipc]
                            assert args.aug_num == 1

                            if args.aug:
                                    img_syn, lab_syn = number_sign_augment(img_syn, lab_syn)

                            if args.dsa:
                                seed = int(time.time() * 1000) % 100000
                                img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                                img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                            output_real = net(img_real, embed=True, last=args.embed_last).detach()
                            output_syn = net(img_syn, embed=True, last=args.embed_last)

                            loss_c += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

                            logits_syn = net(img_syn)
                            metrics[image_sign] += F.cross_entropy(logits_syn, lab_syn).detach().item()
                            acc_avg[image_sign].add(logits_syn.detach(), lab_syn)
                            
                            syn_ce_loss = 0
                            if args.syn_ce:
                                weight_i = net_acc.value()[0] if net_acc.n != 0 else 0
                                syn_ce_loss += (F.cross_entropy(logits_syn, lab_syn) * weight_i)

                                loss_c += (syn_ce_loss * args.ce_weight)

                            optimizer_img.zero_grad()
                            loss_c.backward()
                            optimizer_img.step()

                            loss += loss_c.item()
                else:
                    raise NotImplemented()
                
                loss_avg = loss / num_classes

                shuffled_net_index = list(range(len(net_list)))
                random.shuffle(shuffled_net_index)
                for j in range(min(args.fetch_net_num, len(shuffled_net_index))):
                    training_net_idx = shuffled_net_index[j]
                    net_train = net_list[training_net_idx]
                    net_train.train()
                    optimizer_net_train = optimizer_list[training_net_idx]
                    acc_meter_net_train = acc_meters[training_net_idx]
                    for i in range(args.model_train_steps):
                        img_real_, lab_real_ = loader_real.sample()
                        real_logit = net_train(img_real_)
                        syn_cls_loss = criterion(real_logit, lab_real_)
                        optimizer_net_train.zero_grad()
                        syn_cls_loss.backward()
                        optimizer_net_train.step()

                        acc_meter_net_train.add(real_logit.detach(), lab_real_)


            if it%10 == 0:
                print('%s iter = %04d, loss = syn:%.4f, net_list size = %s, metrics = syn:%.4f, syn acc = syn:%.4f' % (get_time(), it, \
                    loss_avg, str(len(net_list)), \
                    metrics['syn'], acc_avg['syn'].value()[0] if acc_avg['syn'].n!=0 else 0))

            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))


    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))



if __name__ == '__main__':
    main()


