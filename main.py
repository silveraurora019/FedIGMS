import logging
import torch
import os
import numpy as np
import random
import argparse
import copy
from pathlib import Path

from utils import set_for_logger
from dataloaders import build_dataloader
from loss import DiceLoss, JointLoss
import torch.nn.functional as F
from nets import build_model

# 导入新的聚合器
from aggregator_mixed_sim import MixedSimAggregator
# (mixed_info_geo_similarity 会被 aggregator_mixed_sim 自动导入)


@torch.no_grad()
def get_client_features(local_models, dataloaders, device):
    """
    从所有客户端的验证数据加载器中提取特征（假定使用 UNet_pro）。
    --- 修改：现在提取 z 和 shadow ---
    """
    client_feats_list = []
    for model, loader in zip(local_models, dataloaders):
        model.eval()
        all_z = []
        all_shadow_flat = [] # <--- 新增
        
        try:
            for x, target in loader: # 迭代整个验证集
                x = x.to(device)
                # 假设是 UNet_pro，它返回 (output, z, shadow)
                _, z, shadow = model(x) # <--- 接收 shadow
                
                # --- 将 shadow 展平 ---
                # shadow 的形状是 [B, C, H, W]，z 的形状是 [B, D]
                # 我们用平均池化将其变为 [B, C]
                shadow_flat = F.adaptive_avg_pool2d(shadow, (1, 1)).view(shadow.shape[0], -1)
                # --- 展平结束 ---
                
                all_z.append(z.cpu())
                all_shadow_flat.append(shadow_flat.cpu()) # <--- 存储
            
            if len(all_z) > 0:
                client_z = torch.cat(all_z, dim=0)
                client_shadow = torch.cat(all_shadow_flat, dim=0)
                
                # --- 关键：拼接特征 ---
                combined_features = torch.cat([client_z, client_shadow], dim=1)
                client_feats_list.append(combined_features)
                # --- 拼接结束 ---
                
            else:
                logging.warning(f"一个客户端的验证加载器为空。")
                client_feats_list.append(torch.empty(0, 1)) # 添加一个带无效维度的空张量
        except Exception as e:
             logging.error(f"提取特征时出错: {e}")
             client_feats_list.append(torch.empty(0, 1))
             
    # 检查是否所有客户端都成功提取了特征
    if not client_feats_list or any(f.shape[0] == 0 or f.dim() != 2 for f in client_feats_list):
        logging.error("未能从一个或多个客户端提取有效特征。中止相似度计算。")
        return None

    # 检查特征维度是否一致
    try:
        feat_dim = client_feats_list[0].shape[1]
        if not all(f.shape[1] == feat_dim for f in client_feats_list if f.shape[0] > 0):
            logging.warning("客户端之间的特征维度不匹配。")
            # 过滤掉维度不匹配的
            client_feats_list = [f for f in client_feats_list if f.dim() == 2 and f.shape[1] == feat_dim]
            if not client_feats_list:
                logging.error("特征维度检查后没有剩余客户端。")
                return None
    except IndexError:
        logging.error("特征列表为空。")
        return None

    return client_feats_list


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--data_root', type=str, required=False, 
                        default="E:/A_Study_Materials/Dataset/Prostate", 
                        help="Data directory (指向 fundus 或 prostate 的 .npy 文件夹)")
    parser.add_argument('--dataset', type=str, default='prostate', 
                        help="Dataset type: 'fundus' (4 站点) 或 'prostate' (6 站点)")
    
    # 强制使用 unet_pro，因为两种方法都需要 z 特征
    parser.add_argument('--model', type=str, default='unet_pro', help='Model type (unet or unet_pro). Required by MI-Sim and UFT.')

    parser.add_argument('--rounds', type=int, default=200, help='number of maximum communication round')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')

    parser.add_argument('--log_dir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--save_dir', type=str, required=False, default="./weights/", help='Log directory path')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="L2 regularization strength")
    parser.add_argument('--batch-size', type=int, default=8, help='input batch size for training (default: 64)')
    parser.add_argument('--experiment', type=str, default='experiment_mi_uft', help='Experiment name')

    parser.add_argument('--test_step', type=int, default=1)
    
    # --- FedIGMS (MI-Sim) 参数 ---
    parser.add_argument('--rad_gamma', type=float, default=1.0, help='Gamma for RAD similarity')
    parser.add_argument('--mine_hidden', type=int, default=128, help='Hidden layer size for MINE estimator')
    parser.add_argument('--lr_mine', type=float, default=1e-3, help='Learning rate for MINE estimator')
    parser.add_argument('--alpha_init', type=float, default=0.5, help='Initial alpha value for mixed similarity')
    
    # --- FedU (UFT) 参数 ---
    parser.add_argument('--lambda_uft', type=float, default=0.1, 
                        help='Weight for the UFT regularizer (lambda_2 in paper)')
    parser.add_argument('--uft_beta_u', type=float, default=1.0, 
                        help='Uncertainty tax coefficient (beta_u) for UFT')

    # --- 组合算法参数 ---
    parser.add_argument('--sim_start_round', type=int, default=5, help='Round to start using similarity aggregation')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate for the model')

    args = parser.parse_args()
    return args

# (communication, train, test 函数保持不变)
def communication(server_model, models, client_weights):
    with torch.no_grad():
        device = next(server_model.parameters()).device
        if not isinstance(client_weights, torch.Tensor):
            client_weights = torch.tensor(client_weights, dtype=torch.float32, device=device)
        else:
            client_weights = client_weights.to(device)
        for key in server_model.state_dict().keys():
            temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32, device=device)
            for client_idx in range(len(client_weights)):
                temp += client_weights[client_idx] * models[client_idx].state_dict()[key].to(device)
            server_model.state_dict()[key].data.copy_(temp)
    return server_model

def train(cid, model, dataloader, device, optimizer, epochs, loss_func):
    model.train()
    is_unet_pro = model.__class__.__name__ == 'UNet_pro'
    for epoch in range(epochs):
        train_acc = 0.
        loss_all = 0.
        if len(dataloader) == 0:
            logging.warning(f"Client {cid} training dataloader is empty.")
            continue
        for x, target in dataloader:
            x = x.to(device)
            target = target.to(device)
            if is_unet_pro:
                output, _, _ = model(x) 
            else:
                output = model(x)
            optimizer.zero_grad()
            loss = loss_func(output, target)
            loss_all += loss.item()
            train_acc += DiceLoss().dice_coef(output, target).item()
            loss.backward()
            optimizer.step()
        
        if len(dataloader) > 0:
            avg_loss = loss_all / len(dataloader)
            train_acc = train_acc / len(dataloader)
            logging.info('Client: [%d]  Epoch: [%d]  train_loss: %f train_acc: %f'%(cid, epoch, avg_loss, train_acc))

def test(model, dataloader, device, loss_func):
    model.eval()
    loss_all = 0
    test_acc = 0
    is_unet_pro = model.__class__.__name__ == 'UNet_pro'
    
    if len(dataloader) == 0:
        logging.warning("Test/Val dataloader is empty.")
        return 0.0, 0.0 # 返回 0 避免除零错误

    with torch.no_grad():
        for x, target in dataloader:
            x = x.to(device)
            target = target.to(device)
            if is_unet_pro:
                output, _, _ = model(x)
            else:
                output = model(x)
            loss = loss_func(output, target)
            loss_all += loss.item()
            test_acc += DiceLoss().dice_coef(output, target).item()
        
    acc = test_acc / len(dataloader)
    loss = loss_all / len(dataloader)
    return loss, acc

def _flat_params(model):
    return torch.cat([p.data.float().view(-1).cpu() for p in model.parameters()])

def main(args):
    set_for_logger(args)
    logging.info(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device)

    # 2. 动态定义客户端列表
    if args.dataset == 'fundus':
        clients = ['site1', 'site2', 'site3', 'site4']
    elif args.dataset == 'prostate':
        clients = ['site1', 'site2', 'site3', 'site4', 'site5', 'site6']
    else:
        raise ValueError(f"Unknown client list for dataset: {args.dataset}")

    # 3. build dataset (传入 clients)
    train_dls, val_dls, test_dls, client_weight = build_dataloader(args, clients)
    
    client_weight_tensor = torch.tensor(client_weight, dtype=torch.float32, device=device)

    # 4. build model (传入 clients)
    local_models, global_model = build_model(args, clients, device)

    if args.model != 'unet_pro':
        logging.error("This aggregation method requires 'unet_pro' model.")
        logging.error("Please set --model unet_pro")
        return

    # (初始化 InfoGeo Aggregator)
    try:
        dummy_input = torch.randn(2, 3, 384, 384).to(device) # 假设输入 384x384
        
        # --- 修改：推断新的 feat_dim ---
        _, z_dummy, shadow_dummy = global_model(dummy_input)
        shadow_dummy_flat = F.adaptive_avg_pool2d(shadow_dummy, (1, 1)).view(shadow_dummy.shape[0], -1)
        feat_dim = z_dummy.shape[1] + shadow_dummy_flat.shape[1] # 拼接后的维度
        
        logging.info(f"Detected feature dimension (z + shadow_flat) = {feat_dim}")
        # --- 修改结束 ---
        
    except Exception as e:
        logging.error(f"Could not determine feature dimension: {e}")
        feat_dim = 2048 + 512 # (z_dim + shadow_dim) 假设值
        logging.warning(f"Failed to infer feat_dim, defaulting to {feat_dim}")

    aggregator = MixedSimAggregator(
        feat_dim=feat_dim, 
        rad_gamma=args.rad_gamma,
        mine_hidden=args.mine_hidden,
        lr_mine=args.lr_mine,
        device=device
    )
    with torch.no_grad():
        aggregator.mixer.alpha_param.fill_(torch.logit(torch.tensor(args.alpha_init)))
    logging.info(f"InfoGeoAggregator initialized. Start alpha = {args.alpha_init}")

    # (Loss 和 Optimizer)
    # --- 修改：使用 JointLoss ---
    loss_fun = JointLoss() 
    
    optimizer = []
    for id in range(len(clients)):
        optimizer.append(torch.optim.Adam(local_models[id].parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99)))

    # (训练循环)
    best_dice = 0
    best_dice_round = 0
    best_local_dice = []
    
    # --- (保留) 为验证集反馈添加状态变量 ---
    last_avg_val_dice_tensor = None 

    weight_save_dir = os.path.join(args.save_dir, args.experiment)
    Path(weight_save_dir).mkdir(parents=True, exist_ok=True)
    logging.info('checkpoint will be saved at {}'.format(weight_save_dir))

    
    for r in range(args.rounds):

        logging.info('-------- Commnication Round: %3d --------'%r)

        # 1. 本地训练
        for idx, client in enumerate(clients):
            train(idx, local_models[idx], train_dls[idx], device, optimizer[idx], args.epochs, loss_fun)
            
        temp_locals = copy.deepcopy(local_models)
        
        # 3. 聚合
        S_mix, S_rad, S_mi = None, None, None
        
        if r >= args.sim_start_round:
            logging.info('Calculating Info-Geometry Mixed Similarity...')
            # 3a. 提取特征 (使用验证集)
            # --- (已修改) get_client_features 现在返回拼接特征 ---
            client_feats = get_client_features(temp_locals, val_dls, device) 
            
            if client_feats is None:
                logging.warning("Feature extraction failed. Falling back to FedAvg.")
                aggr_weights = client_weight_tensor
            else:
                # 3b. 计算相似度和权重
                S_mix, S_rad, S_mi, current_alpha = aggregator.compute_similarity_matrix(client_feats)
                logging.info(f'Current Alpha: {current_alpha:.4f}')                        
                aggr_weights = aggregator.weights_from_similarity(S_mix).to(device)
                
                logging.info(f'Aggregator Weights: {aggr_weights.cpu().numpy()}')
                
                if len(aggr_weights) != len(temp_locals):
                    logging.error(f"Aggregator weight count ({len(aggr_weights)}) mismatch client count ({len(temp_locals)}). Falling back to FedAvg.")
                    aggr_weights = client_weight_tensor
            
            # 3c. 执行聚合
            communication(global_model, temp_locals, aggr_weights)

        else: 
            # 3d. 早期轮次使用 FedAvg
            logging.info('Using standard FedAvg aggregation.')
            communication(global_model, temp_locals, client_weight_tensor)

        # 4. 分发全局模型
        global_w = global_model.state_dict()
        for idx, client in enumerate(clients):
            local_models[idx].load_state_dict(global_w)


        if r % args.test_step == 0:
            # 5. 测试 (使用测试集 - 用于最终报告和保存最佳模型)
            avg_loss = []
            avg_dice = []
            for idx, client in enumerate(clients):
                loss, dice = test(local_models[idx], test_dls[idx], device, loss_fun)

                logging.info('client: %s  test_loss:  %f   test_acc:  %f '%(client, loss, dice))
                avg_dice.append(dice)
                avg_loss.append(loss)

            avg_dice_v = sum(avg_dice) / len(avg_dice) if len(avg_dice) > 0 else 0
            avg_loss_v = sum(avg_loss) / len(avg_loss) if len(avg_loss) > 0 else 0
            
            logging.info('Round: [%d]  avg_test_loss: %f avg_test_acc: %f std_test_acc: %f'%(r, avg_loss_v, avg_dice_v, np.std(np.array(avg_dice))))

            # --- (保留) 在验证集上进行评估以获取反馈 ---
            avg_val_dice = []
            for idx, client in enumerate(clients):
                _, val_dice = test(local_models[idx], val_dls[idx], device, loss_fun)
                avg_val_dice.append(val_dice)
            
            current_avg_val_dice_tensor = torch.tensor(avg_val_dice, device=device, dtype=torch.float32)
            logging.info('Round: [%d]  avg_val_acc (for feedback): %f'%(r, current_avg_val_dice_tensor.mean().item()))
            # --- (保留) 结束 ---


            # 6. 更新 Alpha (使用验证集性能提升)
            if r >= args.sim_start_round and S_mix is not None:
                # --- (保留) 使用 last_avg_val_dice_tensor ---
                if last_avg_val_dice_tensor is not None: 
                    try:
                        val_improve = current_avg_val_dice_tensor - last_avg_val_dice_tensor 
                        logging.info(f"Updating alpha with VALIDATION feedback. Improvement: {val_improve.cpu().numpy()}")
                        sig_rad, sig_mi, new_alpha = aggregator.update_alpha_from_feedback(S_rad, S_mi, val_improve)
                        logging.info(f'Alpha update: sig_rad={sig_rad:.4f}, sig_mi={sig_mi:.4f}, new_alpha={new_alpha:.4f}')
                    except Exception as e:
                        logging.warning(f"Could not update alpha: {e}")
                
            last_avg_val_dice_tensor = current_avg_val_dice_tensor
            # --- (保留) 结束 ---

            # 7. 保存最佳模型 (仍然基于测试集性能 avg_dice_v)
            if best_dice < avg_dice_v:
                best_dice = avg_dice_v
                best_dice_round = r
                best_local_dice = avg_dice

                weight_save_path = os.path.join(weight_save_dir, 'best.pth')
                torch.save(global_model.state_dict(), weight_save_path)
            

    logging.info('-------- Training complete --------')
    logging.info('Best avg dice score %f at round %d '%( best_dice, best_dice_round))
    for idx, client in enumerate(clients):
        logging.info('client: %s  test_acc:  %f '%(client, best_local_dice[idx] if idx < len(best_local_dice) else 0.0))


if __name__ == '__main__':
    args = get_args()
    main(args)
