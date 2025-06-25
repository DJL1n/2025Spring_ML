import optimize
import os
from utils.config import EnConfig
from utils.data_loader import data_loader
from utils.model import CausalFlow
from utils.train import Trainer
import torch
import random
import numpy as np
import argparse
import optuna

def objective(trial, gpu_id):
    torch.cuda.set_device(gpu_id)  # 设置进程默认设备
    device = torch.device(f"cuda:{gpu_id}")
    config = EnConfig(
        batch_size=8,
        learning_rate=trial.suggest_loguniform("learning_rate", 1e-6, 1e-3),
        gradient_accumulation_steps=trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4, 8]),
        dropout=0.3,
        use_context=True,
        use_attnFusion=True,
    )
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True

    train_loader, test_loader, val_loader = data_loader(config)
    model = CausalFlow(config,device).to(device)
    trainer = Trainer(config,device)

    max_epochs = 50
    patience = 0
    max_patience = config.early_stop
    best_val_acc = 0.0
    best_val_loss = float('inf')
    prev_loss_model_path = None
    prev_acc_model_path = None

    checkpoint_dir = config.model_save_path
    os.makedirs(checkpoint_dir, exist_ok=True)  # 确保目录存在

    for epoch in range(max_epochs):
        trainer.do_train(model, train_loader)
        eval_results = trainer.do_test(model, val_loader, "VAL")
        val_acc = eval_results['Has0_acc_2']
        val_loss = eval_results['Loss']

        # 基于损失的早停和权重保存
        if val_loss < best_val_loss:
            if prev_loss_model_path and os.path.exists(prev_loss_model_path):
                os.remove(prev_loss_model_path)
            best_val_loss = val_loss
            loss_model_path = os.path.join(checkpoint_dir, f"trial_{trial.number}_loss_{best_val_loss:.4f}.pth")
            torch.save(model.state_dict(), loss_model_path)
            prev_loss_model_path = loss_model_path
            patience = 0  # 损失降低时重置耐心
        else:
            patience += 1  # 损失未降低时增加耐心

        # 仍然保存准确率最高的权重（可选）
        if val_acc > best_val_acc:
            if prev_acc_model_path and os.path.exists(prev_acc_model_path):
                os.remove(prev_acc_model_path)
            best_val_acc = val_acc
            acc_model_path = os.path.join(checkpoint_dir, f"trial_{trial.number}_acc_{best_val_acc:.4f}.pth")
            torch.save(model.state_dict(), acc_model_path)
            prev_acc_model_path = acc_model_path

        if patience >= max_patience:
            break

    return best_val_loss  # 返回最佳损失

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, required=True, help='GPU ID to use (0-3)')
    parser.add_argument('--n_trials', type=int, default=5, help='Number of trials per GPU')
    args = parser.parse_args()

    study = optuna.create_study(
        study_name=f"my_study_gpu{args.gpu_id}",
        storage=f"sqlite:///example_gpu{args.gpu_id}.db",
        direction="minimize",
        load_if_exists=True
    )

    study.optimize(lambda trial: objective(trial, args.gpu_id), n_trials=args.n_trials)

    print(f"GPU {args.gpu_id} 最佳试验:")
    best_trial = study.best_trial
    print(f"最佳trial编号: {best_trial.number}")
    print(f"最佳验证损失: {best_trial.value}")
    print(f"最佳权重文件: {os.path.join('checkpoints', f'trial_{best_trial.number}_loss_{best_trial.value:.4f}.pth')}")
    print("最佳超参数:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")


# python optimize.py --gpu_id 0 --n_trials 5 &
# python optimize.py --gpu_id 1 --n_trials 5 &
# python optimize.py --gpu_id 2 --n_trials 5 &
# python optimize.py --gpu_id 3 --n_trials 5 &
