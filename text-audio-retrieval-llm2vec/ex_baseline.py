import csv
import os
import pickle
import random
from datetime import datetime

import numpy
import pandas
import torch

import data_loader
from utils import directories
from utils import model_utils, data_utils, criterion_utils

torch_home = "/home/xjii0026/ml23_scratch2/xjii0026/torch_cache"
hf_home = "/home/xjii0026/ml23_scratch2/xjii0026/hf_cache"
os.environ["DATA_HOME"] = "/home/xjii0026/ml23_scratch2/xjii0026"
os.environ["TOKENIZERS_PARALLELISM"] = "true"




def get_optimizer_with_layerwise_lr(model, base_lr=2e-5, weight_decay=0.05):
    """
    为不同模块设置不同学习率
    - Audio encoder: base_lr * 0.5
    - Projections: base_lr * 5
    - Text encoder unfrozen: base_lr * 0.1
    """
    param_groups = []
    
    # 1. Audio encoder (小lr)
    param_groups.append({
        'params': model.audio_encoder.parameters(),
        'lr': base_lr * 1,
        'name': 'audio_encoder'
    })
    
    # 2. Audio projection (大lr)
    param_groups.append({
        'params': model.audio_proj.parameters(),
        'lr': base_lr * 1,
        'name': 'audio_proj'
    })
    
    # 3. Text projection (大lr)
    param_groups.append({
        'params': model.text_proj.parameters(),
        'lr': base_lr * 1,
        'name': 'text_proj'
    })
    
    # 4. Text encoder解冻的层 (极小lr，如果有的话)
    unfrozen_params = []
    for param in model.text_encoder.parameters():
        if param.requires_grad:
            unfrozen_params.append(param)
    
    if unfrozen_params:
        param_groups.append({
            'params': unfrozen_params,
            'lr': base_lr * 0.1,
            'name': 'text_encoder_unfrozen'
        })
        print(f"✅ Text encoder: {len(unfrozen_params)} params will be fine-tuned")
    
    # 创建optimizer
    optimizer = torch.optim.AdamW(
        param_groups, 
        lr=base_lr,  # 这个会被每个group的lr覆盖
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
        amsgrad=False
    )
    
    # 打印学习率信息
    print("\n" + "="*70)
    print("Learning Rate Configuration:")
    for group in optimizer.param_groups:
        num_params = sum(p.numel() for p in group['params'])
        print(f"  {group['name']:25s}: lr={group['lr']:.2e}, params={num_params:,}")
    print("="*70 + "\n")
    
    return optimizer




def get_config():
    return {
        "trial": "audiocaps_llm2vec_mixed_NewProjectLayer_4unfrozen_tau01_differlr",  # ← 改个名字，区别于原来的 baseline
        "output_dir": "/home/xjii0026/ml23/xjii0026/my_work/code/author/text-audio-retrieval-llm2vec/outputs",

        "random_seed": None,
        "train": "audiocaps",
        "test": "audiocaps",
        "ckpt": None,
        
        # ✅ LLM2Vec 配置（V100 32GB）
        "freeze_text_encoder": True,  # 冻结 LLM2Vec base model
        "batch_size": 32,             # V100 32GB 可以用 16
        "lr": 2e-5,
        "weight_decay": 0.05,                   

        "start_epoch": 0,
        "max_epochs": 25,
        "rampdown_stop": 20,

        "loss": {
            "tau": 0.1,
            "target": "mixed",
            "target_func": "logistic",
            "audio_modality": False,
            "text_modality": False,
            'threshold': 0.7,
            'alpha': 25.0,
            'beta': 20.0,
        }
    }


def run():
    config = get_config()
    print(config)

    # 路径验证
    print(f"DATA_HOME: {os.environ['DATA_HOME']}")
    print(f"Output dir: {config['output_dir']}")

    # Initialize random seed
    random_seed = config.get("random_seed", None)
    if random_seed is None:
        random_seed = numpy.random.randint(2 ** 9)

    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"[Init] Set random_seed: {random_seed}")
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    numpy.random.seed(random_seed)
    random.seed(random_seed)

    trial_dir = os.path.join(config['output_dir'], config['trial'])
    directories.make_if_not_exists(trial_dir)

    # ✅ 修改：传入 freeze_text_encoder 参数
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "[Init] Build model...")
    model = model_utils.init_model(freeze_text_encoder=config.get('freeze_text_encoder', True))
    
    # ✅ 添加：打印模型参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    # Restore model from checkpoint
    ckpt = config.get("ckpt", None)
    if ckpt is not None and os.path.exists(ckpt):
        model = model_utils.restore(model, ckpt)
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f"[Init] Restore model from checkpoint: {ckpt}")

    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "[Init] Load data...")
    train_ds = data_utils.get_data_set(config['train'], 'train')
    val_ds = data_utils.get_data_set(config['test'], 'val')
    test_ds = data_utils.get_data_set(config['test'], 'test')

    print(f'Training set size: {len(train_ds)}')
    print(f'Validation set size: {len(val_ds)}')
    print(f'Test set size: {len(test_ds)}')

    # ✅ 修改：使用 config['batch_size']
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "[Init] Create dataloaders...")
    train_dl = data_loader.get_train_data_loader(train_ds, batch_size=config['batch_size'], targets=None)
    val_dl = data_loader.get_eval_data_loader(val_ds, shuffle=True, distributed=False)
    test_dl = data_loader.get_eval_data_loader(test_ds, shuffle=True, distributed=False)

    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "[Init] Create learning objective...")
    objective = criterion_utils.CELoss(**config['loss'])

    # ✅ 修改：只优化可训练参数
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "[Init] Create optimizer...")
    optimizer = get_optimizer_with_layerwise_lr(
        model, 
        base_lr=config.get('lr', 2e-5),
        weight_decay=config.get('weight_decay', 0.05)
    )

    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "[Init] Create gradient scaler...")
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = config['start_epoch']
    max_epoch = config['max_epochs']

    train_results, val_results, test_results = [], [], []
    val_losses, best_epoch = [], start_epoch

    # Reload epoch results
    report_fpath = os.path.join(trial_dir, "results.csv")
    if os.path.exists(report_fpath):
        val_losses = pandas.read_csv(report_fpath)['val_loss'].tolist()

    # Train model
    for epoch in range(start_epoch, max_epoch):
        train_state = model_utils.train(model, train_dl, objective, optimizer, scaler, epoch,
                                        config['max_epochs'], config['rampdown_stop'])
        train_results.append(train_state)

        val_state = model_utils.test(model, val_dl, objective)
        val_results.append(val_state)

        test_state = model_utils.test(model, test_dl, objective)
        test_results.append(test_state)

        val_losses.append(val_state['test_loss'])
        if val_losses[-1] <= min(val_losses):  # Save checkpoint
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'[Output] Save model checkpoint at Epoch {epoch}...')
            best_epoch = epoch
            ckp_fpath = os.path.join(trial_dir, "best_loss_checkpoint.pt")
            torch.save((model.state_dict(), optimizer.state_dict(), scaler.state_dict()), ckp_fpath)

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), f'[Output] Save model checkpoint at Epoch {epoch}...')
        ckp_fpath = os.path.join(trial_dir, "checkpoint.pt")
        torch.save((model.state_dict(), optimizer.state_dict(), scaler.state_dict()), ckp_fpath)

        # Report epoch results
        report_fpath = os.path.join(trial_dir, "results.csv")
        is_exist = os.path.exists(report_fpath)
        with open(report_fpath, mode='a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)

            if not is_exist:
                writer.writerow(['epoch',
                                 'train_loss',
                                 'val_loss',
                                 'val_t2a_R1', 'val_t2a_R5', 'val_t2a_R10', 'val_t2a_mAP',
                                 'val_a2t_R1', 'val_a2t_R5', 'val_a2t_R10', 'val_a2t_mAP',
                                 'test_loss',
                                 'test_t2a_R1', 'test_t2a_R5', 'test_t2a_R10', 'test_t2a_mAP',
                                 'test_a2t_R1', 'test_a2t_R5', 'test_a2t_R10', 'test_a2t_mAP'])

            writer.writerow([epoch,
                             f"{train_state:.4f}",
                             f"{val_state['test_loss']:.4f}",
                             f"{val_state['audio_retrieval']['R1']:.4f}",
                             f"{val_state['audio_retrieval']['R5']:.4f}",
                             f"{val_state['audio_retrieval']['R10']:.4f}",
                             f"{val_state['audio_retrieval']['mAP']:.4f}",
                             f"{val_state['text_retrieval']['R1']:.4f}",
                             f"{val_state['text_retrieval']['R5']:.4f}",
                             f"{val_state['text_retrieval']['R10']:.4f}",
                             f"{val_state['text_retrieval']['mAP']:.4f}",
                             f"{test_state['test_loss']:.4f}",
                             f"{test_state['audio_retrieval']['R1']:.4f}",
                             f"{test_state['audio_retrieval']['R5']:.4f}",
                             f"{test_state['audio_retrieval']['R10']:.4f}",
                             f"{test_state['audio_retrieval']['mAP']:.4f}",
                             f"{test_state['text_retrieval']['R1']:.4f}",
                             f"{test_state['text_retrieval']['R5']:.4f}",
                             f"{test_state['text_retrieval']['R10']:.4f}",
                             f"{test_state['text_retrieval']['mAP']:.4f}"])

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              f'[Info] Epoch {epoch} -- train_loss: {train_state:.4f}, val_loss: {val_losses[-1]:.4f}')

    # After training
    best_test = test_results[best_epoch]
    print(f'[Test] Best results -- epoch: {best_epoch}',
          f"t2a_R1: {best_test['audio_retrieval']['R1']:.4f}",
          f"t2a_R5: {best_test['audio_retrieval']['R5']:.4f}",
          f"t2a_R10: {best_test['audio_retrieval']['R10']:.4f}",
          f"t2a_mAP: {best_test['audio_retrieval']['mAP']:.4f}",
          f"a2t_R1: {best_test['text_retrieval']['R1']:.4f}",
          f"a2t_R5: {best_test['text_retrieval']['R5']:.4f}",
          f"a2t_R10: {best_test['text_retrieval']['R10']:.4f}",
          f"a2t_mAP: {best_test['text_retrieval']['mAP']:.4f}")

    # Save results
    res_fpath = os.path.join(trial_dir, "results.pkl")
    with open(res_fpath, "wb") as f:
        pickle.dump({"train": train_results, "val": val_results, "test": test_results}, f)
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "[Output] Save training results to", res_fpath)


if __name__ == "__main__":
    run()