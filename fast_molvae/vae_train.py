import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

import math, random
import numpy as np
import argparse
import pickle as pickle
import os
from fast_jtnn import *
import rdkit
from tqdm import tqdm

def get_total_batches(data_folder, batch_size):
    """计算所有 .pkl 文件的总 batch 数"""
    total_batches = 0
    data_files = [fn for fn in os.listdir(data_folder) if fn.endswith('.pkl')]
    for fn in data_files:
        fn = os.path.join(data_folder, fn)
        with open(fn, 'rb') as f:
            data = pickle.load(f)
        # 计算该文件的 batch 数，去掉不足 batch_size 的部分
        num_samples = len(data)
        full_batches = num_samples // batch_size  # 整数除法，丢弃不足一个 batch 的部分
        total_batches += full_batches
    return total_batches

def main_vae_train(train,
                   vocab,
                   save_dir,
                   load_epoch=0,
                   hidden_size=256,
                   batch_size=64,
                   latent_size=64,
                   depthT=20,
                   depthG=3,
                   lr=1e-3,
                   clip_norm=50.0,
                   beta=0.0,
                   step_beta=0.002,
                   max_beta=1.0,
                   warmup=12000,
                   epoch=10,
                   anneal_rate=0.9,
                   anneal_iter=12000, 
                   kl_anneal_iter=2000,
                   print_iter=50, 
                   save_iter=5000):
    
    # 初始化词汇表
    vocab = [x.strip("\r\n ") for x in open(vocab)] 
    vocab = Vocab(vocab)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # print(torch.cuda.get_device_name(0))
    model = JTNNVAE(vocab, int(hidden_size), int(latent_size), int(depthT), int(depthG)).to(device)
    

    # 创建保存目录
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    # 初始化 TensorBoard
    writer = SummaryWriter(save_dir + '/runs')

    # 初始化模型参数
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)
    
    # 加载已有模型
    if load_epoch > 0:
        model.load_state_dict(torch.load(save_dir + "/model.iter-" + str(load_epoch)))

    # 设置优化器和调度器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, anneal_rate)
    # scheduler.step()

    # 定义参数和梯度范数计算函数
    param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
    grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

    total_step = load_epoch
    beta = beta
    meters = np.zeros(4)  # [kl_div, wacc, tacc, sacc]
    
    # 动态计算总步数
    batches_per_epoch = get_total_batches(train, batch_size)
    total_steps = batches_per_epoch * epoch

    # 使用 tqdm 显示 step 进度
    with tqdm(total=total_steps, desc="Training", unit="step") as pbar:
        for e in range(epoch):
            loader = MolTreeFolder(train, vocab, batch_size)
            for batch in loader:
                total_step += 1
                try:
                    model.zero_grad()
                    loss, kl_div, wacc, tacc, sacc = model(batch, beta)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    optimizer.step()
                    if total_step % anneal_iter == 0:  # 然后更新学习率
                        scheduler.step()
                except Exception as e:
                    # 保存异常时的状态并跳过
                    torch.save(model.state_dict(), save_dir + f"/crash_step_{total_step}.pth")
                    pbar.update(1)
                    continue

                # 累积指标
                meters += np.array([kl_div, wacc * 100, tacc * 100, sacc * 100])

                # 每 print_iter 步记录到 TensorBoard
                if total_step % print_iter == 0:
                    meters /= print_iter
                    writer.add_scalar('Loss/Total', loss.item(), total_step)
                    writer.add_scalar('Loss/KL', meters[0], total_step)
                    # 假设重构损失 = 总损失 - beta * KL，若模型不同需调整
                    recon_loss = loss.item() - beta * meters[0]
                    writer.add_scalar('Loss/Reconstruction', recon_loss, total_step)
                    writer.add_scalar('Accuracy/Word', meters[1], total_step)
                    writer.add_scalar('Accuracy/Topo', meters[2], total_step)
                    writer.add_scalar('Accuracy/Assm', meters[3], total_step)
                    writer.add_scalar('Norm/Param', param_norm(model), total_step)
                    writer.add_scalar('Norm/Grad', grad_norm(model), total_step)
                    writer.add_scalar('Hyperparams/Beta', beta, total_step)
                    writer.add_scalar('Hyperparams/LR', scheduler.get_last_lr()[0], total_step)
                    meters *= 0

                # 保存模型：每 5000 step
                # if total_step % save_iter == 0:
                #     torch.save(model.state_dict(), save_dir + f"/model_step_{total_step}.pth")
                
                # 保存检查点：每 10 个 epoch
                if total_step % (batches_per_epoch * 10) == 0 and total_step > 0:  # 避免在 0 step 保存
                    torch.save({
                        'step': total_step,
                        'epoch': e,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'beta': beta
                    }, save_dir + f"/checkpoint_epoch_{e}_step_{total_step}.pth")

                # 调度逻辑
                if total_step % anneal_iter == 0:
                    scheduler.step()
                if total_step % kl_anneal_iter == 0 and total_step >= warmup:
                    beta = min(max_beta, beta + step_beta)

                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({'loss': loss.item(), 'beta': beta})

    # 训练结束保存最终模型
    torch.save(model.state_dict(), save_dir + f"/final_model.pth")
    writer.close()
    return model

if __name__ == '__main__':
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--vocab', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--load_epoch', type=int, default=0)

    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--latent_size', type=int, default=56)
    parser.add_argument('--depthT', type=int, default=20)
    parser.add_argument('--depthG', type=int, default=3)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--clip_norm', type=float, default=50.0)
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--step_beta', type=float, default=0.002)
    parser.add_argument('--max_beta', type=float, default=1.0)
    parser.add_argument('--warmup', type=int, default=12000)

    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--anneal_rate', type=float, default=0.9)
    parser.add_argument('--anneal_iter', type=int, default=15000)
    parser.add_argument('--kl_anneal_iter', type=int, default=2000)
    parser.add_argument('--print_iter', type=int, default=50)
    parser.add_argument('--save_iter', type=int, default=5000)

    args = parser.parse_args()
    
    main_vae_train(args.train,
                   args.vocab,
                   args.save_dir,
                   args.load_epoch,
                   args.hidden_size,
                   args.batch_size,
                   args.latent_size,
                   args.depthT,
                   args.depthG,
                   args.lr,
                   args.clip_norm,
                   args.beta,
                   args.step_beta,
                   args.max_beta,
                   args.warmup,
                   args.epoch, 
                   args.anneal_rate,
                   args.anneal_iter, 
                   args.kl_anneal_iter,
                   args.print_iter, 
                   args.save_iter)