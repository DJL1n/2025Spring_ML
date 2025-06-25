import torch
from utils.train import Trainer
from utils.config import EnConfig
from utils.model import Model
from utils.data_loader import data_loader
import os
from tqdm import tqdm

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1. 设置模型配置
config = EnConfig(
    batch_size=1,           # 设置为1以逐个处理切片
    learning_rate=5e-6,     # 与训练时一致
    seed=1,                 # 与训练时一致
    dataset_name='test',    # 设置为 'test' 以加载新测试数据集
    num_hidden_layers=5,    # 与训练时一致
    use_context=True,       # 与训练时一致
    use_attnFusion=True     # 与训练时一致
)

# 2. 获取测试数据加载器
test_loader = data_loader(config)

# 3. 实例化模型
model = Model(config).to(device)

# 4. 权重文件路径
weight_paths = [
    'checkpoint/RH_acc_mosi_1_0.8734.pth',
    'checkpoint/RH_loss_mosi_1_2.6443.pth'
]

# 5. 实例化 Trainer
trainer = Trainer(config)

# 6. 定义按视频评估的测试函数
def do_test_per_video(trainer, model, data_loader, mode, weight_name):
    model.eval()
    metrics = trainer.metrics  # 获取评估指标计算方法

    print(f"\n=== Evaluating with weights: {weight_name} ===")

    # 存储每个视频的预测和真实标签
    video_predictions = {}
    video_true_labels = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing Clips"):
            # 获取批次数据
            text_inputs = batch["text_tokens"].to(device)
            text_mask = batch["text_masks"].to(device)
            text_context_inputs = batch["text_context_tokens"].to(device)
            text_context_mask = batch["text_context_masks"].to(device)
            audio_inputs = batch["audio_inputs"].to(device)
            audio_mask = batch["audio_masks"].to(device)
            audio_context_inputs = batch["audio_context_inputs"].to(device)
            audio_context_mask = batch["audio_context_masks"].to(device)
            targets = batch["targets"].to(device).view(-1, 1)
            video_id = batch["video_id"][0]  # 获取视频ID

            # 前向传播
            if config.use_context:
                outputs = model(
                    text_inputs, text_mask, text_context_inputs, text_context_mask,
                    audio_inputs, audio_mask, audio_context_inputs, audio_context_mask
                )
            else:
                outputs = model(text_inputs, text_mask, audio_inputs, audio_mask)

            # 收集预测和真实标签
            if video_id not in video_predictions:
                video_predictions[video_id] = {m: [] for m in trainer.tasks}
                video_true_labels[video_id] = []
            for m in trainer.tasks:
                pred = outputs[m].cpu()
                video_predictions[video_id][m].append(pred)
            video_true_labels[video_id].append(targets.cpu())

    # 对每个视频计算评估指标
    for video_id in video_predictions:
        print(f"\nVideo ID: {video_id}")
        true_tensor = torch.cat(video_true_labels[video_id], dim=0)
        for m in trainer.tasks:
            pred_list = video_predictions[video_id][m]
            pred_tensor = torch.cat(pred_list, dim=0)
            results = metrics(pred_tensor, true_tensor)
            result_str = " ".join([f"{k}: {v:.4f}" for k, v in results.items()])
            print(f"{m}: {result_str}")

# 7. 对每个权重文件执行测试
for weight_path in weight_paths:
    # 加载权重
    model.load_state_dict(torch.load(weight_path))
    print(f"Loaded weights from {weight_path}")

    # 执行测试
    do_test_per_video(trainer, model, test_loader, "TEST", os.path.basename(weight_path))
