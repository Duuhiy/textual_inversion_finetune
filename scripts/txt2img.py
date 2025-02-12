import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from torch import optim
from torchvision.transforms import transforms
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
import torch.nn.functional as F

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from model.models.protonet import ProtoNet


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_fix_dataset(args, batch_id, transform):
    batch_size = args.eval_way * (args.eval_shot + args.eval_query)
    save_path = "./fix"
    test_dataset = torch.zeros((batch_size, 3, 84, 84))
    batch_paths = []
    batch_path = os.path.join(save_path, str(batch_id + 1))
    for j in range(batch_size):
        img_dir_path = os.path.join(batch_path, str(j))
        img_path = os.path.join(img_dir_path, os.listdir(img_dir_path)[0])
        batch_paths.append(img_path)
        img = transform(Image.open(img_path).convert('RGB'))
        # print(img.shape)
        test_dataset[j] = img
    return test_dataset, batch_paths

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default="/data/pretrained_models/ldm/text2img-large/model.ckpt", 
        help="Path to pretrained ldm text2img model")

    parser.add_argument(
        "--embedding_path", 
        type=str, 
        help="Path to a pre-trained embedding manager checkpoint")
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "--backbone_class",
        type=str,
        default="Res12")
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--eval_way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--eval_shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--eval_query', type=int, default=15)
    parser.add_argument('--temperature', type=float, default=64)

    parser.add_argument(
        "--epoch",
        type=int,
        default=10,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument("--batch_id",
        type=int,
        default=0)

    opt = parser.parse_args()


    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval_with_tokens.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, opt.ckpt_path)  # TODO: check path
    model.embedding_manager.load(opt.embedding_path)
    model.embedding_manager.string_to_param_dict["*"].requires_grad = True
    print(model.embedding_manager.string_to_param_dict["*"])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model = model.to(device)
    model = model.cuda()

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt


    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    # base_count = 0
    base_count = len(os.listdir(sample_path))

    all_samples=list()
    opt.query = opt.n_samples * opt.n_iter
    opt.eval_query = opt.n_samples * opt.n_iter
    opt.sa = True
    classifier = ProtoNet(opt)
    classifier.load_state_dict(torch.load("/home/stu/dyh/vir/gen/feat_aug/checkpoints/MiniImageNet-ProtoNet-Res12-05w05s15q-Pre-DIS/40_0.5_lr0.0002mul10_step_T164.0T232.0_b0.1_bsz100-NoAug-sa-epoch100-p0.5/max_acc.pth")['params'], strict=False)
    for name, parameter in classifier.named_parameters():
        parameter.requires_grad = False
    # classifier = classifier.to(device)
    classifier = classifier.cuda()
    optimizer = optim.Adam(
        [{'params': model.embedding_manager.string_to_param_dict["*"]}],
        lr=0.0002,
        # weight_decay=args.weight_decay, do not use weight_decay here
    )
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(10),
        gamma=0.2
    )
    transform = transforms.Compose(
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
    resize = transforms.Resize((84, 84))
    # 读取原始batch
    support, _ = load_fix_dataset(opt, opt.batch_id, transform)
            # for x_sample in x_samples_ddim:
            #     x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
            #     Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.jpg"))
            #     base_count += 1
            # all_samples.append(x_samples_ddim)
    # support = support.to(device)
    support = support[:opt.eval_way * opt.eval_shot].cuda()
    print(support.shape)
    query = torch.zeros((opt.eval_query, 3, 84, 84))
    with model.ema_scope():
        uc = None
        if opt.scale != 1.0:
            uc = model.get_learned_conditioning(opt.n_samples * [""])
        for n in trange(opt.n_iter, desc="Sampling"):
            c = model.get_learned_conditioning(opt.n_samples * [prompt])
            shape = [4, opt.H // 8, opt.W // 8]
            samples_ddim, x0s = sampler.sample(S=opt.ddim_steps,
                                               conditioning=c,
                                               batch_size=opt.n_samples,
                                               shape=shape,
                                               verbose=False,
                                               unconditional_guidance_scale=opt.scale,
                                               unconditional_conditioning=uc,
                                               eta=opt.ddim_eta)
            # [5,3,256,256]
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            query = resize(x_samples_ddim)
    data = torch.cat((support, query), dim=0)
    print(str(data.shape))
    # data = data.cuda()
    classifier.train()
    for i in range(opt.epoch):
        logits, reg_logits = classifier(data)
        label = torch.tensor([int(opt.name)] * opt.eval_query).cuda()
        print(label.shape)
        print(logits.shape)

        loss = F.cross_entropy(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    sava_path = os.path.join(os.path.dirname(opt.embedding_path), "fine_tune")
    os.makedirs(sava_path, exist_ok=True)
    idx = len(os.listdir(sava_path))
    model.embedding_manager.save(os.path.join(sava_path, "embeddings" + str(idx) + ".pt"))

    # # additionally, save as grid
    # grid = torch.stack(all_samples, 0)
    # grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    # grid = make_grid(grid, nrow=opt.n_samples)
    #
    # # to image
    # grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    # Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}.jpg'))
    #
    # print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")
