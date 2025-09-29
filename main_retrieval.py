from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import datetime
from os.path import join, exists

import torch

from models.tokenization_clip import SimpleTokenizer as ClipTokenizer
from dataloaders.data_dataloaders import DATALOADER_DICT
from models.modeling import AllGather, MyModel
from models.optimization import BertAdam
from utils.metric_logger import MetricLogger
from utils.metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim

from utils.comm import is_main_process, synchronize
from utils.logger import setup_logger

import warnings

warnings.filterwarnings("ignore")

allgather = AllGather.apply

global logger


def get_args(description='Text-Video Retrieval.'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_train", type=int, default=0, help="Whether to run training.")
    parser.add_argument("--do_eval", type=int, default=0, help="Whether to run evaluation.")

    parser.add_argument("--datatype", type=str, default="msrvtt", help="Point the dataset to finetune.")
    parser.add_argument('--anno_path', type=str, default='MSRVTT/anns', help='annotation path')
    parser.add_argument('--video_path', type=str, default='MSRVTT/videos', help='video path')
    parser.add_argument('--data_path', type=str, default='MSRVTT/', help='data pickle file path')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--workers', type=int, default=8, help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--coef_lr', type=float, default=1e-3, help='coefficient for bert branch.')
    parser.add_argument("--warmup_proportion", type=float, default=0.1,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
    parser.add_argument('--weight_decay', type=float, default=0.2, help='weight decay')
    parser.add_argument('--epochs', type=int, default=5, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=32, help='batch size eval')

    parser.add_argument('--max_words', type=int, default=24, help='max text token number')
    parser.add_argument('--max_frames', type=int, default=12, help='max key frames')
    parser.add_argument('--video_framerate', type=int, default=1, help='framerate to sample video frame')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')

    parser.add_argument("--device", default='cpu', type=str, help="cpu/cuda")
    parser.add_argument("--world_size", default=1, type=int, help="distribted training")
    # maybe you should set as --local_rank based on system.
    parser.add_argument(
        "--local_rank", "--local-rank",
        dest="local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help="local rank for DistributedDataParallel"
    )    
    parser.add_argument("--distributed", default=0, type=int, help="multi machine DDP")

    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument("--output_dir", type=str, default=None, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    # base_encoder = ["ViT-B/32", "ViT-B/16"]
    parser.add_argument("--base_encoder", type=str, default="ViT-B/32", help="Choose a CLIP version")
    parser.add_argument('--agg_module', type=str, default="seqTransf", choices=["None", "seqLSTM", "seqTransf"],
                        help="choice a feature aggregation module for video.")
    parser.add_argument('--interaction', type=str, default='wti', help="interaction type for retrieval.")
    parser.add_argument('--num_hidden_layers', type=int, default=4)
    parser.add_argument("--init_model", type=str, default=None, required=False, help="Initial model.")
    # Different datasets have different split_batch settings, e.g, MSRVTT=25, DiDeMo=17
    parser.add_argument('--split_batch', type=int, default=32, help='test dataset split')
    parser.add_argument('--alpha', type=float, default=0.5, help='hyper-parameters alpha')
    parser.add_argument('--beta', type=float, default=0.5, help='hyper-parameters beta')
    parser.add_argument('--gamma', type=float, default=0.01, help='hyper-parameters gamma')
    parser.add_argument('--resume_from', type=str, default=None,
                            help="The checkpoint file path to resume training from.")
    args = parser.parse_args()
    return args


def set_seed_logger(args):
    global logger
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    use_cuda = torch.cuda.is_available()
    using_dist = ("RANK" in os.environ and "WORLD_SIZE" in os.environ)

    if use_cuda:
        args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
    else:
        args.device = torch.device("cpu")
        args.local_rank = 0
    using_dist = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)

    if use_cuda and using_dist and not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()
    else:
        args.world_size = 1

    # if torch.cuda.is_available():
    #     torch.distributed.init_process_group(backend="nccl")
    #     # torch.distributed.init_process_group(backend="gloo")
    #     torch.cuda.set_device(args.local_rank)
    #     args.device = torch.device("cuda", args.local_rank)
    #     args.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    logger.info(f"local_rank: {args.local_rank} world_size: {args.world_size}")

    if args.batch_size % max(1, args.world_size) != 0 or args.batch_size_val % max(1, args.world_size) != 0:
        raise ValueError(
            f"Invalid batch_size/batch_size_val and world_size: "
            f"{args.batch_size}%{args.world_size} and {args.batch_size_val}%{args.world_size}"
        )

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info(f"  <<< {key}: {args.__dict__[key]}")
    return args


def build_model(args):
    model = MyModel(args)
    if args.init_model:
        if not exists(args.init_model):
            raise FileNotFoundError
        model_state_dict = torch.load(args.init_model, map_location='cpu')
        model.load_state_dict(model_state_dict, strict=False)

    model.to(args.device)
    return model


def build_dataloader(args):
    tokenizer = ClipTokenizer()

    # Here you can add val datasets.
    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)
        logger.info("***** Running testing *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))

    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_length)
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num steps = %d", len(train_dataloader))
    else:
        train_dataloader, train_sampler = None, None

    return test_dataloader, train_dataloader, train_sampler


def prep_optimizer(args, model, num_train_optimization_steps, local_rank):
    if hasattr(model, 'module'):
        model = model.module
    lr = args.lr
    coef_lr = args.coef_lr
    weight_decay = args.weight_decay
    warmup_proportion = args.warmup_proportion
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None

    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)
    # Do not DDP package
    # if torch.cuda.is_available():
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
    #                                                       find_unused_parameters=True)
    return optimizer, scheduler, model


def save_model(epoch, args, model, type_name=""):
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name == "" else type_name + ".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file


def reduce_loss(loss, args):
    world_size = args.world_size
    if world_size < 2:
        return loss
    with torch.no_grad():
        torch.distributed.reduce(loss, dst=0)
        if torch.distributed.get_rank() == 0:
            loss /= world_size
    return loss


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, max_steps):
    global logger
    global best_score
    global meters

    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    total_loss = 0

    end = time.time()
    for step, batch in enumerate(train_dataloader, start=1):
        global_step += 1
        data_time = time.time() - end

        # if n_gpu == 1:
        #     batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        batch = tuple(t.to(device, non_blocking=True) for t in batch)

        text, text_mask, video, video_mask, idx, _ = batch
        loss = model(text, text_mask, video, video_mask, idx, global_step)

        # if n_gpu > 1:
        #     loss = loss.mean()

        # with torch.autograd.detect_anomaly():
        #     loss.backward()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # optimizer.zero_grad()
        with torch.no_grad():
            reduced_l = loss.detach().clone()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(reduced_l, op=torch.distributed.ReduceOp.SUM)
                reduced_l /= args.world_size

        if hasattr(model, 'module'):
            torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            logit_scale = model.module.clip.logit_scale.exp().item()
        else:
            torch.clamp_(model.clip.logit_scale.data, max=np.log(100))
            logit_scale = model.clip.logit_scale.exp().item()

        batch_time = time.time() - end
        end = time.time()

        # reduced_l = reduce_loss(loss, args)
        meters.update(time=batch_time, data=data_time, loss=float(reduced_l))

        eta_seconds = meters.time.global_avg * (max_steps - global_step)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if (global_step % log_step == 0 or global_step == 1) and is_main_process():
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}, ",
                        "epoch: {epoch}/{max_epoch}, ",
                        "iteration: {step}/{len}/{iteration}/{max_iteration}, ",
                        "{meters}",
                        "lr: {lr}, ",
                        "logit: {logit_scale}, ",
                        "memory: {memory:.2f}GB",
                    ]
                ).format(
                    eta=eta_string,
                    epoch=epoch + 1,
                    max_epoch=args.epochs,
                    step=step,
                    len=len(train_dataloader),
                    iteration=global_step,
                    max_iteration=max_steps,
                    meters=str(meters),
                    lr="/".join([str('%.9f' % itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                    logit_scale=logit_scale,
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0,
                )
            )
    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step


def _run_on_single_gpu(model, t_mask_list, s_feat_list, w_feat_list, v_mask_list, f_feat_list, p_feat_list, split_batch=8):

    sim_matrix = []

    batch_t_mask = torch.split(t_mask_list, split_batch)
    batch_s_feat = torch.split(s_feat_list, split_batch)
    batch_w_feat = torch.split(w_feat_list, split_batch)
    batch_v_mask = torch.split(v_mask_list, split_batch)

    batch_f_feat = list(zip(*[torch.split(f, split_batch) for f in f_feat_list]))
    batch_p_feat = list(zip(*[torch.split(p, split_batch) for p in p_feat_list]))

    with torch.no_grad():
        for idx1, (t_mask, s_feat, w_feat) in tqdm(enumerate(zip(batch_t_mask, batch_s_feat, batch_w_feat))):
            each_row = []
            for idx2, (v_mask, f_feat, p_feat) in enumerate(zip(batch_v_mask, batch_f_feat, batch_p_feat)):
                logits = model.get_similarity_logits(t_mask, s_feat, w_feat, v_mask, list(f_feat), list(p_feat))
                logits = logits.cpu().detach().numpy()
                each_row.append(logits)
            each_row = np.concatenate(each_row, axis=-1)
            sim_matrix.append(each_row)

    return sim_matrix

def eval_epoch(args, model, test_dataloader, device):
    global test_dataset

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # multi_sentence_ = False
    # cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    # if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
    #         and test_dataloader.dataset.multi_sentence_per_video:
    #     multi_sentence_ = True
    #     cut_off_points_ = test_dataloader.dataset.cut_off_points
    #     sentence_num_ = test_dataloader.dataset.sentence_num
    #     video_num_ = test_dataloader.dataset.video_num
    #     cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    # if multi_sentence_:
    #     logger.warning("Eval under the multi-sentence per video clip setting.")
    #     logger.warning("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()

    local_s_feat_list, local_w_feat_list, local_t_mask_list = [], [], []
    local_f_feat_list, local_p_feat_list, local_v_mask_list = [], [], []
    local_inds_list = []

    if is_main_process():
        logger.info(f"Rank {args.local_rank} starting feature extraction...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader, disable=not is_main_process()):
            batch = tuple(t.to(device) for t in batch)
            text, text_mask, video, video_mask, inds, index = batch

            s_feat, w_feat, f_feat, p_feat = model.get_text_video_feat(text, text_mask, video, video_mask)

            local_s_feat_list.append(s_feat)
            local_w_feat_list.append(w_feat)
            local_t_mask_list.append(text_mask)
            local_f_feat_list.append(f_feat)
            local_p_feat_list.append(p_feat)
            local_v_mask_list.append(video_mask)
            local_inds_list.append(inds)

    synchronize()
    def cat_(lst): return torch.cat(lst, dim=0) if len(lst) > 0 else None
    s_feat_local = cat_(local_s_feat_list)
    w_feat_local = cat_(local_w_feat_list)
    t_mask_local = cat_(local_t_mask_list)
    v_mask_local = cat_(local_v_mask_list)
    inds_local   = cat_(local_inds_list)
    # === Step 2: 使用all_gather在所有GPU上同步完整的特征张量 ===
    # 这样每个GPU都拥有了整个测试集的数据，为并行计算做准备
    if len(local_f_feat_list) > 0:
        f_feat_by_layer_local = list(zip(*local_f_feat_list))  # list[tensor(B, F, D)]
        p_feat_by_layer_local = list(zip(*local_p_feat_list))
    else:
        f_feat_by_layer_local, p_feat_by_layer_local = [], []
    s_feat_all = allgather(s_feat_local, args)
    w_feat_all = allgather(w_feat_local, args)
    t_mask_all = allgather(t_mask_local, args)
    v_mask_all = allgather(v_mask_local, args)
    inds_all   = allgather(inds_local, args).view(-1)  # [N]

    f_feat_all_layers = [allgather(torch.cat(layer, dim=0), args) for layer in f_feat_by_layer_local] if f_feat_by_layer_local else []
    p_feat_all_layers = [allgather(torch.cat(layer, dim=0), args) for layer in p_feat_by_layer_local] if p_feat_by_layer_local else []
    order = torch.argsort(inds_all)
    s_feat_all = s_feat_all[order]
    w_feat_all = w_feat_all[order]
    t_mask_all = t_mask_all[order]
    v_mask_all = v_mask_all[order]
    f_feat_all_layers = [x[order] for x in f_feat_all_layers]
    p_feat_all_layers = [x[order] for x in p_feat_all_layers]

    synchronize()
    # --- Text Features ---
    # s_feat_gathered = allgather(torch.cat(local_s_feat_list, dim=0), args)
    # w_feat_gathered = allgather(torch.cat(local_w_feat_list, dim=0), args)
    # t_mask_gathered = allgather(torch.cat(local_t_mask_list, dim=0), args)
    
    # --- Video Features ---
    # f_feat and p_feat are lists of tensors, we need to gather them layer by layer
    # f_feat_all_layers = list(zip(*[item for sublist in local_f_feat_list for item in sublist]))
    # f_feat_gathered_layers = [allgather(torch.cat(layer_feats, dim=0), args) for layer_feats in f_feat_all_layers]
    
    # p_feat_all_layers = list(zip(*[item for sublist in local_p_feat_list for item in sublist]))
    # p_feat_gathered_layers = [allgather(torch.cat(layer_feats, dim=0), args) for layer_feats in p_feat_all_layers]
    # if local_f_feat_list:
    #     f_feat_by_layer = list(zip(*local_f_feat_list))
    #     f_feat_gathered_layers = [allgather(torch.cat(layer_feats, dim=0), args) for layer_feats in f_feat_by_layer]
    # else:
    #     f_feat_gathered_layers = []

    # if local_p_feat_list:
    #     p_feat_by_layer = list(zip(*local_p_feat_list))
    #     p_feat_gathered_layers = [allgather(torch.cat(layer_feats, dim=0), args) for layer_feats in p_feat_by_layer]
    # else:
    #     p_feat_gathered_layers = []
    # f_feat_all_layers = list(zip(*local_f_feat_list)) if len(local_f_feat_list) > 0 else []
    # f_feat_gathered_layers = [allgather(torch.cat(layer_feats, dim=0), args) for layer_feats in f_feat_all_layers] if len(f_feat_all_layers) > 0 else []

    # p_feat_all_layers = list(zip(*local_p_feat_list)) if len(local_p_feat_list) > 0 else []
    # p_feat_gathered_layers = [allgather(torch.cat(layer_feats, dim=0), args) for layer_feats in p_feat_all_layers] if len(p_feat_all_layers) > 0 else []
    # v_mask_gathered = allgather(torch.cat(local_v_mask_list, dim=0), args)
    
    # if is_main_process():
    #     logger.info("All features gathered across all GPUs.")
    #     logger.info(f"Total text features gathered: {s_feat_gathered.shape[0]}")
    #     logger.info(f"Total video features gathered: {v_mask_gathered.shape[0]}")

    # # === Step 3: 分布式计算相似度矩阵 ===
    # # 每个GPU计算一部分文本与所有视频的相似度
    
    # # 获取当前进程的rank
    # world_size = torch.distributed.get_world_size()
    # rank = torch.distributed.get_rank()

    # # 计算每个GPU需要处理的文本数量
    # total_texts = s_feat_gathered.shape[0]
    # texts_per_gpu = (total_texts + world_size - 1) // world_size
    # start_idx = rank * texts_per_gpu
    # end_idx = min(start_idx + texts_per_gpu, total_texts)

    # sim_matrix_chunk = []
    # if start_idx < end_idx:
    #     t_mask_chunk = t_mask_gathered[start_idx:end_idx]
    #     s_feat_chunk = s_feat_gathered[start_idx:end_idx]
    #     w_feat_chunk = w_feat_gathered[start_idx:end_idx]
        
    #     if rank == 0:
    #         logger.info("Starting distributed similarity matrix computation...")
        
    #     with torch.no_grad():
    #         sim_chunk = _run_on_single_gpu(model, t_mask_chunk, s_feat_chunk, w_feat_chunk, 
    #                                        v_mask_gathered, f_feat_gathered_layers, p_feat_gathered_layers, 
    #                                        args.split_batch)
    #         sim_matrix_chunk = np.concatenate(tuple(sim_chunk), axis=0)

    # synchronize()

    # === Step 4: 收集所有GPU计算的部分结果 ===
    # 使用 all_gather_object 收集numpy数组
    # make sure each rank provides a numpy array (possibly empty) so the gathered list has consistent types
    # if not isinstance(sim_matrix_chunk, np.ndarray):
    #     total_videos = v_mask_gathered.shape[0]
    #     sim_matrix_chunk = np.zeros((0, total_videos), dtype=np.float32)
    # gathered_chunks = [None] * world_size
    # torch.distributed.all_gather_object(gathered_chunks, sim_matrix_chunk)
    
    # === Step 5: 在主进程上拼接矩阵并计算指标 ===
    if is_main_process():
        logger.info("Similarity computation finished. Gathering and finalizing results.")
        
        # 过滤掉空块并拼接
        gathered_chunks = [chunk for chunk in gathered_chunks if chunk is not None and chunk.shape[0] > 0]
        if not gathered_chunks:
            logger.error("No similarity chunks were gathered. Cannot compute metrics.")
            return 0.0
            
        sim_matrix = _run_on_single_gpu(
            model,
            t_mask_all, s_feat_all, w_feat_all,
            v_mask_all, f_feat_all_layers, p_feat_all_layers,
            split_batch=args.split_batch
        )
        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
        logger.info("Final similarity matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        
        multi_sentence_ = False
        if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') and test_dataloader.dataset.multi_sentence_per_video:
            multi_sentence_ = True
            cut_off_points_ = test_dataloader.dataset.cut_off_points
            cut_off_points_ = [itm - 1 for itm in cut_off_points_]
            logger.warning("Eval under the multi-sentence per video clip setting.")
            logger.warning(f"Sentence num: {test_dataloader.dataset.sentence_num}, Video num: {test_dataloader.dataset.video_num}")
        
        if multi_sentence_:
            logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
            cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
            max_length = max([e_ - s_ for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_)])
            sim_matrix_new = []
            for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
                sim_matrix_new.append(
                    np.concatenate((sim_matrix[s_:e_], np.full((max_length - e_ + s_, sim_matrix.shape[1]), -np.inf)), axis=0)
                )
            sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
            logger.info("after reshape, sim matrix size: {} x {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))
            tv_metrics = tensor_text_to_video_metrics(sim_matrix)
            vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
        else:
            tv_metrics = compute_metrics(sim_matrix)
            vt_metrics = compute_metrics(sim_matrix.T)
        
        tv_metrics['RSum'] = tv_metrics['R1'] + tv_metrics['R5'] + tv_metrics['R10']
        logger.info("Text-to-Video: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@Sum: {:.1f} - MdR: {:.1f} - MnR: {:.1f}".format(
            tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['RSum'], tv_metrics['MR'], tv_metrics['MeanR']
        ))
        vt_metrics['RSum'] = vt_metrics['R1'] + vt_metrics['R5'] + vt_metrics['R10']
        logger.info("Video-to-Text: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@Sum: {:.1f} - MdR: {:.1f} - MnR: {:.1f}".format(
            vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['RSum'], vt_metrics['MR'], vt_metrics['MeanR']
        ))
        return tv_metrics['R1']
    
    # 非主进程必须返回一个值，以保持函数签名一致
    return 0.0
    # logger.info("Model begins to testing...")

    # ids_t, batch_mask_t, batch_feat_s, batch_feat_w = [], [], [], []
    # ids_v, batch_mask_v, batch_feat_f, batch_feat_p = [], [], [], []

    # with torch.no_grad():
    #     tic = time.time()
    #     if multi_sentence_:
    #         total_video_num = 0
    #         for batch in tqdm(test_dataloader):
    #             batch = tuple(t.to(device) for t in batch)
    #             text, text_mask, video, video_mask, inds, index = batch

    #             b, *_t = video.shape
    #             s_feat, w_feat = model.get_text_feat(text, text_mask)
    #             ids_t.append(inds)
    #             batch_mask_t.append(text_mask)
    #             batch_feat_s.append(s_feat)
    #             batch_feat_w.append(w_feat)

    #             s_, e_ = total_video_num, total_video_num + b
    #             filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

    #             if len(filter_inds) > 0:
    #                 vide, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]

    #                 f_feat, p_feat = model.get_video_feat(video, video_mask)
    #                 ids_v.append(inds)
    #                 batch_feat_f.append(f_feat)
    #                 batch_mask_v.append(video_mask)
    #                 batch_feat_p.append(p_feat)
    #             total_video_num += b

    #         ids_t = torch.cat(ids_t, dim=0).squeeze()
    #         ids_v = torch.cat(ids_v, dim=0).squeeze()
    #         batch_mask_t = torch.cat(batch_mask_t, dim=0)
    #         batch_feat_s = torch.cat(batch_feat_s, dim=0)
    #         batch_feat_w = torch.cat(batch_feat_w, dim=0)
    #         batch_mask_v = torch.cat(batch_mask_v, dim=0)
    #         batch_feat_f = list(zip(*batch_feat_f))
    #         batch_feat_p = list(zip(*batch_feat_p))
    #         batch_feat_f = [torch.cat(layer_feats, dim=0) for layer_feats in batch_feat_f]
    #         batch_feat_p = [torch.cat(layer_feats, dim=0) for layer_feats in batch_feat_p]
    #     else:
    #         for batch in tqdm(test_dataloader):
    #             batch = tuple(t.to(device) for t in batch)
    #             text, text_mask, video, video_mask, inds, index = batch
    #             s_feat, w_feat, f_feat, p_feat = model.get_text_video_feat(text, text_mask, video, video_mask)
    #             ids_t.append(inds)
    #             ids_v.append(inds)
    #             batch_mask_t.append(text_mask)
    #             batch_mask_v.append(video_mask)
    #             batch_feat_s.append(s_feat)
    #             batch_feat_w.append(w_feat)
    #             batch_feat_f.append(f_feat)
    #             batch_feat_p.append(p_feat)
    #         ids_t = torch.cat(ids_t, dim=0).squeeze()
    #         ids_v = torch.cat(ids_v, dim=0).squeeze()
    #         batch_mask_t = torch.cat(batch_mask_t, dim=0)
    #         batch_feat_s = torch.cat(batch_feat_s, dim=0)
    #         batch_feat_w = torch.cat(batch_feat_w, dim=0)
    #         batch_mask_v = torch.cat(batch_mask_v, dim=0)
    #         batch_feat_f = list(zip(*batch_feat_f))
    #         batch_feat_p = list(zip(*batch_feat_p))
    #         batch_feat_f = [torch.cat(layer_feats, dim=0) for layer_feats in batch_feat_f]
    #         batch_feat_p = [torch.cat(layer_feats, dim=0) for layer_feats in batch_feat_p]
    # toc1 = time.time()
    # with torch.no_grad():
    #     sim_matrix = _run_on_single_gpu(model, batch_mask_t, batch_feat_s, batch_feat_w, batch_mask_v, batch_feat_f,
    #                                     batch_feat_p, args.split_batch)
    #     sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
    # toc2 = time.time()
    # if multi_sentence_:
    #     logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
    #     cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
    #     max_length = max([e_ - s_ for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_)])
    #     sim_matrix_new = []
    #     for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
    #         sim_matrix_new.append(
    #             np.concatenate((sim_matrix[s_:e_], np.full((max_length - e_ + s_, sim_matrix.shape[1]), -np.inf)),
    #                            axis=0))
    #     sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
    #     logger.info("after reshape, sim matrix size: {} x {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1],
    #                                                                       sim_matrix.shape[2]))
    #     tv_metrics = tensor_text_to_video_metrics(sim_matrix)
    #     vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
    #     toc3 = time.time()
    #     logger.info(
    #         "time profile: feat {:.1f}s match {:.5f}s metrics {:.5f}s".format(toc1 - tic, toc2 - toc1, toc3 - toc2))
    #     tv_metrics['RSum'] = tv_metrics['R1'] + tv_metrics['R5'] + tv_metrics['R10']
    #     logger.info(
    #         "Text-to-Video: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@Sum: {:.1f} - MdR: {:.1f} - MnR: {:.1f}".
    #         format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['RSum'], tv_metrics['MR'],
    #                tv_metrics['MeanR']))
    #     vt_metrics['RSum'] = vt_metrics['R1'] + vt_metrics['R5'] + vt_metrics['R10']
    #     logger.info(
    #         "Video-to-Text: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@Sum: {:.1f} - MdR: {:.1f} - MnR: {:.1f}".
    #         format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['RSum'], vt_metrics['MR'],
    #                vt_metrics['MeanR']))
    #     return tv_metrics['R1']

    # else:
    #     logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
    #     tv_metrics = compute_metrics(sim_matrix)
    #     vt_metrics = compute_metrics(sim_matrix.T)
    #     logger.info('Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))
    #     logger.info('[end] compute_metrics')
    #     toc3 = time.time()
    #     logger.info(
    #         "time profile: feat {:.1f}s match {:.5f}s metrics {:.5f}s".format(toc1 - tic, toc2 - toc1, toc3 - toc2))
    #     tv_metrics['RSum'] = tv_metrics['R1'] + tv_metrics['R5'] + tv_metrics['R10']
    #     logger.info(
    #         "Text-to-Video: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@Sum: {:.1f} - MdR: {:.1f} - MnR: {:.1f}".
    #         format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['RSum'], tv_metrics['MR'],
    #                tv_metrics['MeanR']))
    #     vt_metrics['RSum'] = vt_metrics['R1'] + vt_metrics['R5'] + vt_metrics['R10']
    #     logger.info(
    #         "Video-to-Text: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@Sum: {:.1f} - MdR: {:.1f} - MnR: {:.1f}".
    #         format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['RSum'], vt_metrics['MR'],
    #                vt_metrics['MeanR']))
    #     return tv_metrics['R1']


def main():
    global logger
    global best_score
    global meters

    meters = MetricLogger(delimiter="")
    args = get_args()
    if args.resume_from and exists(args.resume_from):
        args.output_dir = os.path.dirname(args.resume_from)
    else:
        args.output_dir = os.path.join(args.output_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))

    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
    import logging
    logger = logging.getLogger('Model')
    args = set_seed_logger(args)
    synchronize()
    logger = setup_logger('Model', args.output_dir, args.local_rank)


    model = build_model(args)

    test_dataloader, train_dataloader, train_sampler = build_dataloader(args)

    if args.do_train:
        tic = time.time()
        max_steps = len(train_dataloader) * args.epochs
        _max_steps = len(train_dataloader) * 5

        # breakpoint setting
        start_epoch = 0
        global_step = 0
        best_score = 0.00001
        best_output_model_file = "None"

        # do not DDP
        optimizer, scheduler, _ = prep_optimizer(args, model, _max_steps, args.local_rank)

        if args.resume_from and exists(args.resume_from):
            logger.info(f"Resuming training from checkpoint: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=args.device)
            
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_step']
            
            model.load_state_dict(checkpoint['state_dict'])
            
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            best_score = checkpoint['best_score']
            
            logger.info(f"Resumed from epoch {start_epoch}, global step {global_step}, best score {best_score}")

            logger.info("Performing an evaluation on the resumed model before continuing training...")
            if torch.cuda.is_available():
                temp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
            eval_epoch(args, temp_model, test_dataloader, args.device)
            synchronize()
            del temp_model 
        
        if torch.cuda.is_available():
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                              find_unused_parameters=True)
        logger.info("Model begins to training...")        

        for epoch in range(args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            synchronize()
            if epoch == -1:
                torch.cuda.empty_cache()
                logger.info("Model zero-shot of text-to-video retrieval")
                R1 = eval_epoch(args, model, test_dataloader, args.device)
                logger.info("Zero-shot text-to-video retrieval of R1 is: {:.4f}".format(R1))

            torch.cuda.empty_cache()
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader,
                                               args.device, args.world_size, optimizer,
                                               scheduler, global_step, max_steps)
            synchronize()

            if args.local_rank == 0:
                checkpoint_state = {
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_score': best_score,
                }
                output_checkpoint_file = join(args.output_dir, f"checkpoint_epoch_{epoch}.pth")
                torch.save(checkpoint_state, output_checkpoint_file)
                logger.info(f"Checkpoint for epoch {epoch} (training complete) saved to {output_checkpoint_file}")             

            synchronize()

            torch.cuda.empty_cache()
            R1 = eval_epoch(args, model, test_dataloader, args.device)
            torch.cuda.empty_cache()
            synchronize()
            if args.local_rank == 0:
                if best_score <= R1:
                    best_score = R1
                    best_output_model_file = output_checkpoint_file

                    best_model_weights = torch.load(best_output_model_file, map_location='cpu')['state_dict']
                    torch.save(best_model_weights, join(args.output_dir, 'best.pth'))

                logger.info("The best model is from checkpoint: {}, current R1: {:.4f}, best R1: {:.4f}".format(best_output_model_file, R1, best_score))
            
            synchronize()

        toc = time.time() - tic
        training_time = time.strftime("%Hh %Mmin %Ss", time.gmtime(toc))
        logger.info("*" * 20 + '\n' + f'training finished with {training_time}' + "*" * 20 + '\n')
        if torch.cuda.is_available():
            torch.distributed.barrier()
        best_state = torch.load(join(args.output_dir, 'best.pth'), map_location='cpu')
        if hasattr(model, 'module'):
            model.module.load_state_dict(best_state, strict=False)
        else:
            model.load_state_dict(best_state, strict=False)
        # if args.local_rank == 0:
        #     model.load_state_dict(torch.load(best_output_model_file, map_location='cpu'), strict=False)
        # if torch.cuda.is_available():
        #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
        #                                                       find_unused_parameters=True)

        torch.cuda.empty_cache()
        eval_epoch(args, model, test_dataloader, args.device)
        synchronize()

    elif args.do_eval:
        R1 = eval_epoch(args, model, test_dataloader, args.device)
        if is_main_process():
            logger.info("Evaluation finished. Text-to-Video R@1: {:.4f}".format(R1))


if __name__ == "__main__":
    main()
