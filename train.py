# train.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from dataset_CL import DataSet
from Loss_function import graph_loss
import warnings
warnings.filterwarnings("ignore")


seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)


parser = argparse.ArgumentParser(description='PyTorch Relationship')
parser.add_argument('--mode', dest="mode", default="PISC_Fine", type=str,
                    help='PISC_Fine, PISC_Coarse')
parser.add_argument('--network', default='network', type=str, help='Network name.')
parser.add_argument('--scene', default=True, type=bool, help='whether scene node exists')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (defult: 4)')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 1)')
parser.add_argument('--print_freq', '-p', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--scale-size', default=256, type=int,
                    help='input size')
parser.add_argument('--crop-size', default=224, type=int,
                    help='crop size')

# use your saved path of the Task1 trained model to replace the xxx
models = {
    'Task1': "/xxx",
}

parser.add_argument('-PMs', '--pretrained_models', default=['Task1'], type=list,
                    help='List of pretrained models.')

# use your save path to replace the xxx
parser.add_argument('--result_path', default='/xxx', type=str, metavar='PATH',
                    help='Path for saving result.')

parser.add_argument('--epochs', dest="epochs", default=200, type=int,
                    help='path for saving result (default: none)')
parser.add_argument('--lr', dest="lr", default=0.0001, type=float,
                    help='Learning rate.')
parser.add_argument('--start_epoch', dest="start_epoch", default=1, type=int,
                    help='')
parser.add_argument('--lr_step', dest='lr_step', default=20, type=int, metavar='N',
                    help='Step to adjust learning rate.')

args = parser.parse_args()

for k, v in sorted(vars(args).items()):
    print(k, ': ', v)


if args.mode == "PISC_Fine":
    data_dir = r"/xxx/image"
    num_class = 6
    train_list = r"/data/data_split"
    test_list = r"/data/data_split"
elif args.mode == "PISC_Coarse":
    data_dir = r"/xxx/image"
    num_class = 3
    train_list = r"/data/data_split"
    test_list = r"/data/data_split"


def vg_collate(data):
    Name = []
    Union = []
    Obj1 = []
    Obj2 = []
    Bpos = []
    Target = []
    Full_im = []
    Img_rel_num = []

    for d in data:
        name, union, obj1, obj2, bpos, target, full_im, img_rel_num = d
        # print(name)
        # print(union.shape)

        Name.append(name)
        Union.append(union)
        Obj1.append(obj1)
        Obj2.append(obj2)
        Bpos.append(bpos)
        Target.append(target)
        Full_im.append(full_im)
        Img_rel_num.append(img_rel_num)

    Union = torch.cat(Union, 0)
    Obj1 = torch.cat(Obj1, 0)
    Obj2 = torch.cat(Obj2, 0)
    Bpos = torch.cat(Bpos, 0)
    Target = torch.cat(Target, 0)
    Full_im = torch.cat(Full_im, 0)
    Full_im = Full_im.view(-1, 3, 224, 224)
    Img_rel_num = torch.cat(Img_rel_num, 0)

    return Name, Union, Obj1, Obj2, Bpos, Target, Full_im, Img_rel_num


def get_test_set(data_dir, test_list):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    crop_size = args.crop_size
    test_data_transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        normalize])
    test_full_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize])
    test_set = DataSet(data_dir, test_list, test_data_transform, test_full_transform)
    test_loader = DataLoader(dataset=test_set, num_workers=args.workers, collate_fn=vg_collate,
                             batch_size=args.batch_size, shuffle=False)
    return test_loader


def get_train_set(data_dir, train_list):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    scale_size = args.scale_size
    crop_size = args.crop_size
    train_data_transform = transforms.Compose([
        transforms.Resize((scale_size, scale_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(crop_size),
        transforms.ToTensor(),
        normalize])
    train_full_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])
    train_set = DataSet(data_dir, train_list, train_data_transform, train_full_transform)
    train_loader = DataLoader(dataset=train_set, num_workers=args.workers, collate_fn=vg_collate,
                              batch_size=args.batch_size, shuffle=False)
    return train_loader, train_set


def validate(val_loader, model):
    model.eval()
    tp, p, r = {}, {}, {}
    for batch_data in tqdm(val_loader):
        name, union, obj1, obj2, bpos, target, full_im, img_rel_num = batch_data
        target = target.cuda()
        union_var = torch.autograd.Variable(union).cuda()
        obj1_var = torch.autograd.Variable(obj1).cuda()
        obj2_var = torch.autograd.Variable(obj2).cuda()
        bpos_var = torch.autograd.Variable(bpos).cuda()
        full_im_var = torch.autograd.Variable(full_im).cuda()
        edge_index = generate_graph(img_rel_num[0])
        if args.scene:
            count = img_rel_num[0] + 1
        else:
            count = img_rel_num[0]
        for rel_num in img_rel_num[1:]:
            edge_index = torch.cat((edge_index, generate_graph(rel_num) + count), dim=1)
            if args.scene:
                count += rel_num + 1
            else:
                count += rel_num

        img_rel_num = torch.autograd.Variable(img_rel_num).cuda()
        edge_index = torch.autograd.Variable(edge_index).cuda()

        target_var = torch.autograd.Variable(target)
        with torch.no_grad():
            output, personPair, person_1, person_2, scene = model(union_var, obj1_var, obj2_var, bpos_var, full_im_var, img_rel_num, edge_index)

        output_f = F.softmax(output, dim=1)
        output_np = output_f.data.cpu().numpy()
        pre = np.argmax(output_np, 1)
        t = target_var.data.cpu().numpy()

        predict_label = np.argmax(output_np, 1)

        true_label = target_var.data.cpu().numpy()
        difference = true_label - predict_label
        sum_diff = 0

        for number in difference:
            if number == 0:
                sum_diff += 1

        for i, item in enumerate(t):
            if item in r:
                r[item] += 1
            else:
                r[item] = 1
            if pre[i] in p:
                p[pre[i]] += 1
            else:
                p[pre[i]] = 1
            if pre[i] == item:
                if item in tp:
                    tp[item] += 1
                else:
                    tp[item] = 1

    return sum_diff / len(true_label)


def softmax_with_temp(logits, temp):

    return F.softmax(logits / temp, dim=1)

def train(train_loader, test_loader, model, criterion, optimizer, model_previous, Res_Loss):
    best_AA = 0
    for i in range(args.start_epoch, args.start_epoch + args.epochs):
        model.train()
        losses = AverageMeter()
        total_losses = AverageMeter()
        top1 = AverageMeter()
        adjust_learning_rate(optimizer, i)
        for j, batch_data in enumerate(tqdm(train_loader)):
            name, union, obj1, obj2, bpos, target, full_im, img_rel_num = batch_data

            target = target.cuda()
            union_var = torch.autograd.Variable(union).cuda()
            obj1_var = torch.autograd.Variable(obj1).cuda()
            obj2_var = torch.autograd.Variable(obj2).cuda()
            bpos_var = torch.autograd.Variable(bpos).cuda()
            full_im_var = torch.autograd.Variable(full_im).cuda()

            edge_index = generate_graph(img_rel_num[0])
            if args.scene:
                count = img_rel_num[0] + 1
            else:
                count = img_rel_num[0]
            for rel_num in img_rel_num[1:]:
                edge_index = torch.cat((edge_index, generate_graph(rel_num) + count), dim=1)
                if args.scene:
                    count += rel_num + 1
                else:
                    count += rel_num

            img_rel_num = torch.autograd.Variable(img_rel_num).cuda()
            edge_index = torch.autograd.Variable(edge_index).cuda()

            target_var = torch.autograd.Variable(target)
            optimizer.zero_grad()
            output_current, pair_now, person_1_now, person_2_now, scene_now = model(union_var, obj1_var, obj2_var, bpos_var, full_im_var, img_rel_num, edge_index)
            output_soft_current = F.softmax(output_current, 1)

            lossCE = criterion(output_current, target_var)

            output_previous, pair_pre, person_1_pre, person_2_pre, scene_pre = model_previous(union_var, obj1_var, obj2_var, bpos_var, full_im_var, img_rel_num, edge_index)
            output_logit_previous = softmax_with_temp(output_previous, 2)

            # GKD
            graphLoss = graph_loss(3)
            lossGraph = graphLoss(pair_now, person_1_now, person_2_now, scene_now,
                                  pair_pre, person_1_pre, person_2_pre, scene_pre)
            lossLogit = Res_Loss(output_soft_current.log(), output_logit_previous)
            loss = lossCE + lossGraph + lossLogit

            losses.update(loss.item(), union.size(0))
            total_losses.update(loss.item())

            prec1 = accuracy(output_current.data, target)
            top1.update(prec1[0], union.size(0))

            loss.backward()
            optimizer.step()

        print("Train: {}/{}: \tPrec@1 {} ({})\t".format(i, len(train_loader), top1.val, top1.avg))
        losses.reset()
        print("epoch {}\ttotal_loss {:.4f}".format(i, total_losses.avg))
        # print("Losses: CE: {:.4f}, Graph: {:.4f}, Logit: {:.4f}".format(lossCE, lossGraph, lossLogit))

        AA = validate(test_loader, model)

        if best_AA < AA:
            best_AA = AA
            torch.save({
                "epoch": i,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()},
                "{}/Task_2_{}_{:.2%}.pth.tar".format(args.result_path, i, AA))
        total_losses.reset()
        print("validate ====>> AA: {:.2%}".format(AA))


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def generate_graph(rel_num, scene=args.scene):
    if scene:
        numNode = rel_num + 1
    else:
        numNode = rel_num
    edge_index = []
    if numNode != 1:
        for i in range(numNode):
            for j in range(numNode):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        edge_index = edge_index.t().contiguous()
    else:
        edge_index = [[], []]
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        edge_index = edge_index.contiguous()
    return edge_index


def init_network(net, num_class):
    # Initialize the network.
    if net == 'network':
        from network import fusion
        model = fusion(num_class)
        for m in model.pp.parameters():
            m.requires_grad = False
        for m in model.pp.bboxes.parameters():
            m.requires_grad = True
        for m in model.pp.fc_pair.parameters():
            m.requires_grad = True
        for m in model.pp.fc_A.parameters():
            m.requires_grad = True
        for m in model.pp.fc_B.parameters():
            m.requires_grad = True
        for m in model.pp.fc_scene.parameters():
            m.requires_grad = True
        for m in model.multiattn_intra.parameters():
            m.requires_grad = True
        for m in model.multiattn_inter.parameters():
            m.requires_grad = True
        for m in model.fc_Class.parameters():
            m.requires_grad = True
        params = [
            {"params": model.pp.bboxes.parameters()},
            {"params": model.pp.fc_pair.parameters()},
            {"params": model.pp.fc_A.parameters()},
            {"params": model.pp.fc_B.parameters()},
            {"params": model.pp.fc_scene.parameters()},
            {"params": model.fc_Class.parameters()},
            {"params": model.cls_token},
            {"params": model.multiattn_intra.parameters()},
            {"params": model.multiattn_inter.parameters()},
        ]

    return model, params


if __name__ == '__main__':
    # Create dataloader
    print('====> Creating dataloader...')
    dataPath_train = "PISCF_train_task_2.pkl"
    dataPath_test = "PISCF_test_task_2.pkl"

    train_task_list = os.path.join(train_list, dataPath_train)
    test_task_list = os.path.join(test_list, dataPath_test)

    train_loader, train_set = get_train_set(data_dir, train_task_list)
    test_loader = get_test_set(data_dir, test_task_list)
    print("train_list:", train_task_list)
    print("test_list:", test_task_list)

    # load network
    print('====> Loading the network...')
    model, params = init_network(args.network, num_class)
    optimizer = torch.optim.Adam(params, weight_decay=0.0005)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    Res_Loss = torch.nn.KLDivLoss(reduction='batchmean').cuda()

    model.cuda()

    from network import fusion
    model_previous = fusion(num_class)

    # load model
    for key in args.pretrained_models:
        if key == "Task1":
            pretrain_model_person = torch.load(models[key])
            fg_model_dict = model_previous.state_dict()
            pretrain_model_person = {k.replace('fg.module.', ''): v for k, v in
                                     pretrain_model_person['state_dict'].items()}
            pretrain_model_person = {k: v for k, v in pretrain_model_person.items() if k in fg_model_dict}
            fg_model_dict.update(pretrain_model_person)
            model_previous.load_state_dict(fg_model_dict)
            print("---------Task_1 Loading！---------")
    model_previous.cuda()

    train(train_loader, test_loader, model, criterion, optimizer, model_previous, Res_Loss)
