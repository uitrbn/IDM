import os
import random
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from dc_utils import get_network, get_eval_pool, evaluate_synset, get_time, epoch, DiffAugment, ParamDiffAug, number_sign_augment, parser_bool, downscale
import torchnet
import torch.nn.functional as F
import pickle

from utils import get_dataset as get_dataset_mtt

best_acc = 0

def main():
    global best_acc

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DM', help='DC/DSA/DM')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet_GBN', help='model')
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
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--outer_loop', type=int, default=1, help='outer loop for network update')
    parser.add_argument('--inner_loop', type=int, default=1, help='outer loop for network update')
    parser.add_argument('--eval_interval', type=int, default=100, help='outer loop for network update')
    parser_bool(parser, 'net_train_real', False)
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
    parser.add_argument('--conf_path', type=str, default='', help='outer loop for network update')
    parser.add_argument('--local_num', type=int, default=10, help='outer loop for network update')
    parser.add_argument('--local_weight', type=float, default=0.1, help='outer loop for network update')
    parser_bool(parser, 'syn_ce', False)
    parser.add_argument('--ce_weight', type=float, default=0.1, help='outer loop for network update')
    parser.add_argument('--optim', type=str, default='sgd', help='outer loop for network update')
    parser.add_argument('--train_net_num', type=int, default=2, help='outer loop for network update')
    parser.add_argument('--aug_num', type=int, default=1, help='outer loop for network update')
    parser.add_argument('--net_push_num', type=int, default=1, help='outer loop for network update')

    parser_bool(parser, 'zca', False)
    parser_bool(parser, 'aug', False)

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
    # channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset_mtt(args.dataset, args.data_path, args.batch_real, 'none', args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, 'ConvNet', args.model)

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c=None, n=0): # get random n images from class c
            if c is not None:
                idx_shuffle = np.random.permutation(indices_class[c])[:n]
                return images_all[idx_shuffle]
            else:
                assert n > 0, 'n must be larger than 0'
                indices_flat = [_ for sublist in indices_class for _ in sublist]
                idx_shuffle = np.random.permutation(indices_flat)[:n]
                return images_all[idx_shuffle], labels_all[idx_shuffle]

        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        print("Shape of Image_SYN: {}".format(image_syn.shape))

        if args.init == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                if not args.aug:
                    image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
                else:
                    half_size = im_size[0]//2
                    image_syn.data[c*args.ipc:(c+1)*args.ipc, :, :half_size, :half_size] = downscale(get_images(c, args.ipc), 0.5).detach().data
                    image_syn.data[c*args.ipc:(c+1)*args.ipc, :, half_size:, :half_size] = downscale(get_images(c, args.ipc), 0.5).detach().data
                    image_syn.data[c*args.ipc:(c+1)*args.ipc, :, :half_size, half_size:] = downscale(get_images(c, args.ipc), 0.5).detach().data
                    image_syn.data[c*args.ipc:(c+1)*args.ipc, :, half_size:, half_size:] = downscale(get_images(c, args.ipc), 0.5).detach().data

        elif args.init == 'mtt':
            raise NotImplementedError()
        else:
            print('initialize synthetic data from random noise')

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
            if it in eval_it_pool[1:]:
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
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
                        accs.append(acc_test)
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

                    if it == args.Iteration: # record the final results
                        accs_all_exps[model_eval] += accs

                    # save the checkpoint of synthetic set with best performance\
                    checkpoint_dir = './checkpoints/{}_ipc_{}_aug_{}_model_{}/'.format(args.dataset, args.ipc, args.aug, model_eval)
                    if not os.path.exists(checkpoint_dir):
                        os.mkdir(checkpoint_dir)
                    best_synset_filename = checkpoint_dir + 'acc_{}.pkl'.format(np.mean(accs))
                    if best_acc < np.mean(accs):
                        best_acc = np.mean(accs)
                        with open(best_synset_filename, 'wb') as pkl_file:
                            pickle.dump((image_syn.detach(), label_syn.detach()), pkl_file)
                            print("Saving best synset with accuracy: {}".format(np.mean(accs)))

                ''' visualize and save '''
                save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, exp, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.

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
                for _ in range(args.net_push_num):
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
            else:
                raise NotImplemented()
            train_net_list = [net_list[ind] for ind in net_index_list]
            train_acc_list = [acc_meters[ind] for ind in net_index_list]

            embed_list = [net.module.embed_channel_avg if torch.cuda.device_count() > 1 else net.embed_channel_avg for net in train_net_list]

            for _ in range(args.outer_loop):
                loss_avg = 0
                mtt_loss_avg = 0
                metrics = {'syn': 0, 'real': 0}
                acc_avg = {'syn':torchnet.meter.ClassErrorMeter(accuracy=True)}

                ''' update synthetic data '''
                if 'BN' not in args.model or args.model=='ConvNet_GBN': # for ConvNet
                    for image_sign, image_temp in [['syn', image_syn]]:
                        loss = torch.tensor(0.0).to(args.device)
                        for net_ind in range(len(train_net_list)):
                            net = train_net_list[net_ind]
                            net.eval()
                            embed = embed_list[net_ind]
                            net_acc = train_acc_list[net_ind]
                            for c in range(num_classes):
                                loss_c = torch.tensor(0.0).to(args.device)
                                img_real = get_images(c, args.batch_real)
                                img_syn = image_temp[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))
                                lab_syn = label_syn[c*args.ipc:(c+1)*args.ipc]
                                assert args.aug_num == 1

                                if args.aug:
                                    img_syn, lab_syn = number_sign_augment(img_syn, lab_syn)

                                if args.dsa:
                                    img_real_list = list()
                                    img_syn_list = list()
                                    for aug_i in range(args.aug_num):
                                        seed = int(time.time() * 1000) % 100000
                                        img_real_list.append(DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param))
                                        img_syn_list.append(DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param))
                                    img_real = torch.cat(img_real_list)
                                    img_syn = torch.cat(img_syn_list)
                                
                                if args.ipc == 1 and not args.aug:
                                    logits_real = net(img_real).detach()
                                    loss_real = F.cross_entropy(logits_real, labels_all[indices_class[c]][:img_real.shape[0]], reduction='none')
                                    indices_topk_loss = torch.topk(loss_real, k=2560, largest=False)[1]
                                    img_real = img_real[indices_topk_loss]
                                    metrics['real'] += loss_real[indices_topk_loss].mean().item()

                                output_real = embed(img_real, last=args.embed_last).detach()
                                output_syn = embed(img_syn, last=args.embed_last)

                                loss_c += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

                                logits_syn = net(img_syn)
                                metrics[image_sign] += F.cross_entropy(logits_syn, lab_syn.repeat(args.aug_num)).detach().item()
                                acc_avg[image_sign].add(logits_syn.detach(), lab_syn.repeat(args.aug_num))

                                syn_ce_loss = 0
                                if args.syn_ce:
                                    weight_i = net_acc.value()[0] if net_acc.n != 0 else 0
                                    if args.ipc == 1 and not args.aug:
                                        if logits_syn.argmax() != c:
                                            syn_ce_loss += (F.cross_entropy(logits_syn, lab_syn.repeat(args.aug_num)) * weight_i)
                                    else:
                                        syn_ce_loss += (F.cross_entropy(logits_syn, lab_syn.repeat(args.aug_num)) * weight_i)

                                    loss_c += (syn_ce_loss * args.ce_weight)

                                optimizer_img.zero_grad()
                                loss_c.backward()
                                optimizer_img.step()

                                loss += loss_c.item()

                        if image_sign == 'syn':
                            loss_avg += loss.item()
                else:
                    raise NotImplemented()

                loss_avg /= (num_classes)
                mtt_loss_avg /= (num_classes)
                metrics = {k:v/num_classes for k, v in metrics.items()}

                shuffled_net_index = list(range(len(net_list)))
                random.shuffle(shuffled_net_index)
                for j in range(min(args.fetch_net_num, len(shuffled_net_index))):
                    training_net_idx = shuffled_net_index[j]
                    net_train = net_list[training_net_idx]
                    net_train.train()
                    optimizer_net_train = optimizer_list[training_net_idx]
                    acc_meter_net_train = acc_meters[training_net_idx]
                    for i in range(args.model_train_steps):
                        img_real_, lab_real_ = get_images(c=None, n=args.trained_bs)
                        real_logit = net_train(img_real_)
                        syn_cls_loss = criterion(real_logit, lab_real_)
                        optimizer_net_train.zero_grad()
                        syn_cls_loss.backward()
                        optimizer_net_train.step()
                        acc_meter_net_train.add(real_logit.detach(), lab_real_)

            if it%10 == 0:
                print('%s iter = %04d, loss = syn:%.4f, net_list size = %s, metrics = syn:%.4f/real:%.4f, syn acc = syn:%.4f' % (get_time(), it, \
                    loss_avg, str(len(net_list)), metrics['syn'], metrics['real'], acc_avg['syn'].value()[0] if acc_avg['syn'].n!=0 else 0))

            if it == args.Iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%s_%dipc.pt'%(args.method, args.dataset, args.model, args.ipc)))


    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))


if __name__ == '__main__':
    main()


