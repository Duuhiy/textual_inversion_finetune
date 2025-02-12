import os.path
import random
import time
import os.path as osp
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from einops import rearrange
import shutil
import json
from model.utils import read_and_parse_file

from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer,
)
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval,
)
from tensorboardX import SummaryWriter
from collections import deque
from tqdm import tqdm
from torchvision import transforms

class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model, self.para_model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)
        self.cmd0 = "python main.py --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume /data/models/ldm/text2img-large/model.ckpt --gpus 0, "
        self.cmd1 = 'python scripts/txt2img.py --ddim_eta 0.0 --n_samples 5 --scale 10.0 --ddim_steps 50 --ckpt_path /data/models/ldm/text2img-large/model.ckpt --embedding_path '
        self.transform = transforms.Compose(
            [
                transforms.Resize(92),
                transforms.CenterCrop(84),
                transforms.ToTensor(),
            ]
            +
            [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
            ]
        )
        self.unnormalize = transforms.Normalize(
            np.array([-120.39586422 / 70.68188272, -115.59361427 / 68.27635443, -104.54012653 / 72.54505529])
            , np.array([255.0 / 70.68188272, 255.0 / 68.27635443, 255.0 / 72.54505529]))
        self.bigger = transforms.Resize(224)
        self.idx = [i for i in range(args.shot + args.query)]
        random.shuffle(self.idx)
        file_path = './data/miniimagenet/label.json'  # 文件路径，根据实际情况修改
        self.label2class = read_and_parse_file(file_path)
        self.verb = ["standing", "flying", "swimming", "running", "lying"]
        self.background = [" in the forest", " in the sky", " on the table", " in the river", " on the grassland"]
        self.prompts = ["standing in the forest", "standing on the table", "standing in the river",
                        "standing on the grassland", "flying in the sky", "flying in the forest",
                        "swimming in the river", "running in the forest", "running  on the table",
                        "running in the river", "running on the grassland", "lying in the forest", "lying on the table",
                        "lying in the river", "lying on the grassland"]

    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)

        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)

        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()

        return label, label_aux

    def train(self):
        args = self.args
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        # start FSL training
        label, label_aux = self.prepare_label()
        for epoch in range(1, args.max_epoch + 1):
            self.train_epoch += 1
            self.model.train()
            if self.args.fix_BN:
                self.model.encoder.eval()

            tl1 = Averager()
            tl2 = Averager()
            ta = Averager()

            start_tm = time.time()
            for batch in self.train_loader:
                self.train_step += 1
                # if args.train_aug:
                #     data =self.augment_data(batch, True)
                # else:
                data = batch[0]
                if torch.cuda.is_available():
                    # [100, 3, 84, 84] -> [200, 3, 84, 84]
                    # data = data.cuda()
                    data = data.cuda()
                # else:
                #     data, gt_label = batch[0], batch[1]

                data_tm = time.time()
                self.dt.add(data_tm - start_tm)

                # get saved centers
                logits, reg_logits = self.para_model(data)
                if reg_logits is not None:
                    loss = F.cross_entropy(logits, label)
                    total_loss = loss + args.balance * F.cross_entropy(reg_logits, label_aux)
                else:
                    loss = F.cross_entropy(logits, label)
                    total_loss = F.cross_entropy(logits, label)

                tl2.add(loss)
                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)
                acc = count_acc(logits, label)

                tl1.add(total_loss.item())
                ta.add(acc)

                self.optimizer.zero_grad()
                total_loss.backward()
                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)

                self.optimizer.step()
                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)

                # refresh start_tm
                start_tm = time.time()

            self.lr_scheduler.step()
            self.try_evaluate(epoch)

            print('ETA:{}/{}'.format(
                self.timer.measure(),
                self.timer.measure(self.train_epoch / args.max_epoch))
            )

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')

    def evaluate(self, data_loader):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 2))  # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
            self.trlog['max_acc_epoch'],
            self.trlog['max_acc'],
            self.trlog['max_acc_interval']))
        with torch.no_grad():
            for i, batch in enumerate(data_loader, 1):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i - 1, 0] = loss.item()
                record[i - 1, 1] = acc

        assert (i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:, 0])
        va, vap = compute_confidence_interval(record[:, 1])

        # train mode
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        return vl, va, vap

    def load_fix_dataset(self, task_num=600):
        args = self.args
        batch_size = args.eval_way * (args.eval_shot + args.eval_query)
        save_path = "./fix"
        test_dataset = torch.zeros((task_num, batch_size, 3, 84, 84))
        paths = []
        for i in range(task_num):
            batch_paths = []
            batch_path = os.path.join(save_path, str(i + 1))
            for j in range(batch_size):
                img_dir_path = os.path.join(batch_path, str(j))
                img_path = os.path.join(img_dir_path, os.listdir(img_dir_path)[0])
                batch_paths.append(img_path)
                img = self.transform(Image.open(img_path).convert('RGB'))
                # print(img.shape)
                test_dataset[i, j] = img
            paths.append(batch_paths)
        return test_dataset, paths

    def evaluate_test(self):
        # restore model args
        args = self.args
        # evaluation mode
        # self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        print(len(self.test_loader))
        task_num = 1
        record = np.zeros((task_num, 2))  # loss and acc
        record_old = np.zeros((task_num, 2))  # loss and acc
        test_label_train = torch.arange(args.eval_way, dtype=torch.int16).repeat((args.eval_query + args.eval_shot))
        test_label_train = test_label_train.type(torch.LongTensor)
        test_label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        test_label = test_label.type(torch.LongTensor)
        if torch.cuda.is_available():
            test_label_train = test_label_train.cuda()
            test_label = test_label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
            self.trlog['max_acc_epoch'],
            self.trlog['max_acc'],
            self.trlog['max_acc_interval']))
        # start FSL training
        label, label_aux = self.prepare_label()

        avg_best_epoch = 0
        # if args.fix_batch:
        #     # 一个固定的batch
        #     b0_path = [x.strip() for x in open("./batch0.txt", 'r').readlines()][1:]
        #     H, W = 84, 84
        #     b0 = torch.zeros(((args.shot + args.query) * args.way, 3, H, W))
        #     for j, p in enumerate(b0_path):
        #         img = self.transform(Image.open(p).convert('RGB'))
        #         b0[j] = img
        #     test_dataset, paths = b0.unsqueeze(0), [b0_path]
        # else:
        test_dataset, paths = self.load_fix_dataset(task_num)
        for i, batch in tqdm(enumerate(test_dataset, 1)):
            # with open("./batch0.txt", 'w') as f:
            #     for _, path in enumerate(batch[2]):
            #         f.write(path + "\n")
            # 加载权重、冻结参数
            self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'], strict=False)
            for name, parameter in self.model.named_parameters():
                print(name)
                parameter.requires_grad = True
                # if 'classfier' not in name and 'new' not in name:
                # if 'classfier' not in name and 'layer4.0.conv' not in name:
                #     parameter.requires_grad = False
                # else:
                #     parameter.requires_grad = True
                    # test before fine-tuning

            if args.fc:
                data_test = batch[args.way * args.eval_shot:]
            else:
                data_test = batch

            if torch.cuda.is_available():
                data_test = data_test.cuda()

            with torch.no_grad():
                self.model.eval()
                # data_test = batch[0]
                logits_test = self.model(data_test)
                loss_test = F.cross_entropy(logits_test, test_label)
                acc_test = count_acc(logits_test, test_label)
                record_old[i - 1, 0] = loss_test.item()
                record_old[i - 1, 1] = acc_test


            # 是否通过数据增强，进行微调
            if args.test_aug:
                # instance_embs = self.model(data_test, True)  # [5, 640]
                # emb_dim = instance_embs.size(-1)  # 640
                # support_idx = torch.Tensor(np.arange(args.eval_way * args.eval_shot)).long().view(1, args.eval_shot, args.eval_way)
                # support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))  # [1,5,5,640]
                #
                # # get mean of the support
                # proto = support.mean(dim=1)  # [1,5,640] # Ntask x NK x d
                # proto = F.normalize(proto, dim=2, p=2)
                # proto = proto.squeeze(0)
                # # num_batch = proto.shape[0]
                # # num_proto = proto.shape[1]
                # # num_query = args.aug_num
                # #
                # # # query: (num_batch, num_query, num_proto, num_emb)
                # # # proto: (num_batch, num_proto, num_emb)
                # # # proto.unsqueeze(1): [1,1,5,640]
                # # proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim)  # [1,75,5,640]
                # # proto = proto.contiguous().view(num_batch * num_query, num_proto, emb_dim)  # (Nbatch x Nq, Nk, d) # [75,5,640]
                # proto = np.array(proto.cpu().detach().numpy())
                # np.savetxt("./proto.txt", proto)

                # [25,3,84,84] + [75,3,84,84] -> [100, 3, 84, 84]
                data = self.augment_data(batch, paths[i - 1], i - 1)

                best_epoch = 1
                if torch.cuda.is_available():
                    # [100, 3, 84, 84]
                    data = data.cuda()
                for epoch in range(1, args.meta_epoch + 1):
                    self.model.train()
                    self.train_epoch += 1
                    if self.args.fix_BN:
                        self.model.encoder.eval()

                    tl1 = Averager()
                    tl2 = Averager()
                    ta = Averager()

                    start_tm = time.time()

                    data_tm = time.time()
                    self.dt.add(data_tm - start_tm)

                    # get saved centers
                    logits, reg_logits = self.model(data)
                    if reg_logits is not None:
                        loss = F.cross_entropy(logits, label)
                        total_loss = loss + args.balance * F.cross_entropy(reg_logits, label_aux)
                    else:
                        if args.fc:
                            loss = F.cross_entropy(logits, test_label_train)
                            total_loss = F.cross_entropy(logits, test_label_train)
                        else:
                            loss = F.cross_entropy(logits, label)
                            total_loss = F.cross_entropy(logits, label)

                    tl2.add(loss)
                    forward_tm = time.time()
                    self.ft.add(forward_tm - data_tm)
                    if args.fc:
                        acc = count_acc(logits, test_label_train)
                    else:
                        acc = count_acc(logits, label)

                    tl1.add(total_loss.item())
                    ta.add(acc)

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    backward_tm = time.time()
                    self.bt.add(backward_tm - forward_tm)

                    self.optimizer.step()
                    optimizer_tm = time.time()
                    self.ot.add(optimizer_tm - backward_tm)

                    self.lr_scheduler.step()
                    print('meta fine-tune epoch:{} train acc:{} loss:{}'.format(
                        epoch,
                        acc,
                        total_loss)
                    )
                    # test
                    with torch.no_grad():
                        self.model.eval()
                        logits_test = self.model(data_test)
                        loss_test = F.cross_entropy(logits_test, test_label)
                        acc_test = count_acc(logits_test, test_label)
                        print('test epoch:{} test acc:{} loss:{}'.format(
                            epoch,
                            acc_test,
                            loss_test)
                        )
                        if epoch == 1:
                            record[i - 1, 0] = loss_test.item()
                            record[i - 1, 1] = acc_test
                        elif acc_test > record[i - 1, 1]:
                            record[i - 1, 0] = loss_test.item()
                            record[i - 1, 1] = acc_test
                            best_epoch = epoch
                avg_best_epoch += best_epoch
                print("best epoch:{} test acc:{}".format(best_epoch, record[0, 1]))
            # test
            with torch.no_grad():
                self.model.eval()
                logits_test = self.model(data_test)
                loss_test = F.cross_entropy(logits_test, test_label)
                acc_test = count_acc(logits_test, test_label)
                print('final test test acc:{} loss:{}'.format(
                    acc_test,
                    loss_test)
                )
                record[i - 1, 0] = loss_test.item()
                record[i - 1, 1] = acc_test
        # assert(i == record.shape[0])
        print("average best epoch：" + str(avg_best_epoch))
        for i in range(task_num):
            print('acc before {}, acc after {}\n'.format(record_old[i], record[i]))
        vl, _ = compute_confidence_interval(record[:, 0])
        va, vap = compute_confidence_interval(record[:, 1])
        va_old, vap_old = compute_confidence_interval(record_old[:, 1])

        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap
        self.trlog['test_loss'] = vl

        # print('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
        #         self.trlog['max_acc_epoch'],
        #         self.trlog['max_acc'],
        #         self.trlog['max_acc_interval']))

        print('Test acc={:.4f} + {:.4f}\n'.format(va_old, vap_old))
        print('Test acc={:.4f} + {:.4f}\n'.format(self.trlog['test_acc'], self.trlog['test_acc_interval']))

        return vl, va, vap

    def final_record(self):
        # save the best performance in a txt file

        with open(
                osp.join(self.args.save_path, '{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])),
                'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))

    def augment_data(self, data, path, batch_id):
        # 0~24 分组，存到./images/id
        args = self.args
        idx = self.idx
        label2class = self.label2class
        my_transform = self.transform
        H, W = data.shape[2], data.shape[3]
        # pn = len(self.prompts)
        pn = 1
        aug_num = args.aug_num * pn * args.way
        to_add = torch.zeros((args.query * args.way, 3, H, W))
        rand_data = torch.zeros(((args.shot + args.query) * args.way, 3, H, W))
        # if args.fix_batch:
        #     base_path = os.path.join("./outputs/txt2img-samples", "b0")
        # else:
        #     base_path = "./outputs/txt2img-samples"
        base_path = os.path.join("../mini/5_way_5_shot/txt2img-sample/", str(batch_id))
        # learn_word
        # os.chdir("./augment/textual_inversion")
        # # 清空
        # for i in range(args.way):
        #     img_path = os.path.join("./images", str(i % 5))
        #     if os.path.exists(img_path):
        #         shutil.rmtree(img_path)
        #     os.mkdir(img_path)

        # if args.saa:
        #     x = data.cuda()
        #     w_sa = self.model.sa(x)
        #     x_sa = w_sa * 0.1 + x
        #     for i in range(len(x_sa)):
        #         x_sa[i] = self.unnormalize(x_sa[i])
        #     x_sa = torch.clamp(x_sa, min=0, max=1)
        # for i in range(args.shot * args.way):
        #     if args.saa:
        #         x_sample = 255. * rearrange(x_sa[i].cpu().numpy(), 'c h w -> h w c')
        #         img = Image.fromarray(x_sample.astype(np.uint8))
        #         img = self.bigger(img)
        #         img.save(os.path.join(os.path.join("./images", str(i % 5)), os.path.basename(path[i])))
        #     else:
        #         shutil.copy(path[i], os.path.join("./images", str(i % 5)))
        for i in range(args.way):
            origin_data, gen_data = torch.zeros((args.shot, 3, H, W)).cuda(), torch.zeros((aug_num, 3, H, W)).cuda()

            label = os.path.basename(path[i])[:9]

            out_dir = os.path.join(base_path, str(i))
            # # 清空
            # if os.path.exists(out_dir):
            #     shutil.rmtree(out_dir)
            # os.mkdir(out_dir)

            for j in range(args.shot):
                origin_data[j] = data[j * args.way + i]

            # if args.fix_batch:
            #     cmd0 = self.cmd0 + "--data_root ./images/" + str(i) + " -n " + "b0-" + str(
            #         i) + " --no_test true --init_word " + label2class[label]
            #     print(cmd0)
            #     os.system(cmd0)
            #     cmd1 = self.cmd1 + "./logs/" + "b0-" + str(i) + "/checkpoints/embeddings.pt" + " --outdir ./outputs/txt2img-sample/b0/" + str(i) + " --name " + str(i) + " --n_iter " + str(int(args.aug_num / 5)) + ' --prompt "a photo of *"'
            #     print(cmd1)
            #     os.system(cmd1)
            #     # if args.fine_tune:
            #     # for _ in range(5):
            #     #     os.system(cmd1)
            # else:
            #     cmd0 = self.cmd0 + "--data_root ./images/" + str(i) + " -n " + str(i) + " --no_test true --init_word " + label2class[label]
            #     print(cmd0)
            #     os.system(cmd0)
            #     cmd1 = self.cmd1 + "./logs/" + str(i) + "/checkpoints/embeddings.pt" + " --outdir ./outputs/txt2img-sample/" + str(i) + " --n_iter " + str(int(args.aug_num / 5)) + ' --prompt "a photo of *"'
            #     print(cmd1)
            #     os.system(cmd1)

            # for _, bg in enumerate(self.prompts):
            #     cmd1 = self.cmd1 + "./logs/" + str(i) + "/checkpoints/embeddings.pt" + " --outdir ./outputs/txt2img-sample/" + str(i) + " --n_iter " + str(int(args.aug_num / 5)) + ' --prompt "a photo of * ' + bg + '"'
            #     print(cmd1)
            #     os.system(cmd1)
            # ckptdir = learn_word(str(i), "3,")   # embedding_path
            # generate(i, "a photo of *", os.path.join(ckptdir, "embeddings_gs-2999.pt"), H, W, to_add)

            # 扩充data
            dir_path = os.path.join(out_dir, "samples")
            # print(dir_path)
            dir_list = os.listdir(dir_path)
            for j, file in enumerate(dir_list):
                img_path = os.path.join(dir_path, file)
                img = my_transform(Image.open(img_path).convert('RGB'))
                gen_data[j] = img
            # filter
            choose_data = gen_data
            if aug_num > args.query:
                choose_data = self.filter_augment_data(origin_data, gen_data)
            for j in range(args.query):
                to_add[j * args.way + i] = choose_data[j]
        aug_data = torch.cat((data[:args.way * args.shot], to_add), dim=0)
        # random
        if args.random:
            for i in range(args.way):
                for j in range(args.shot + args.query):
                    rand_data[j * args.way + i] = aug_data[idx[j] * args.way + i]
            return rand_data
        return aug_data

    def filter_augment_data(self, origin_data, gen_data):
        args = self.args
        f = self.para_model(origin_data, True)  # [5, 640]
        emb_dim = f.size(-1)
        # proto
        proto = f.mean(dim=0).unsqueeze(0).expand(gen_data.shape[0], emb_dim)
        fa = self.para_model(gen_data, True)  # [aug_num,640]
        score = - torch.sum((proto - fa) ** 2, 1)
        idx = torch.argsort(score, descending=True)[:args.query]
        to_add = gen_data[idx]
        return to_add


