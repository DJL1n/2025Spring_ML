import torch
import pandas as pd
from utils.model import Model
from utils.config import EnConfig
from utils.data_loader import data_loader
import os

def generate_predictions(model_weights_path, config, csv_path):
    """
    加载保存的模型权重，在测试数据集上生成预测，并更新现有 CSV 文件中的 'prediction' 列。

    参数:
        model_weights_path (str): 保存的模型权重文件路径（例如 'checkpoint/RH_acc_mosi_1_0.5678.pth'）
        config (EnConfig): 与训练设置匹配的配置对象
        csv_path (str): 要更新的现有 CSV 文件路径（例如 'data/Test/label.csv'）
    """
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载测试数据
    test_loader = data_loader(config)

    # 使用提供的配置初始化模型
    model = Model(config).to(device)

    # 加载保存的模型权重
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    # 加载现有的 CSV 文件
    df = pd.read_csv(csv_path)

    # 存储预测结果的列表
    predictions = []

    # 在不计算梯度的情况下生成预测
    with torch.no_grad():
        for batch in test_loader:
            # 将输入移动到设备上
            text_inputs = batch["text_tokens"].to(device)
            text_mask = batch["text_masks"].to(device)
            audio_inputs = batch["audio_inputs"].to(device)
            audio_mask = batch["audio_masks"].to(device)

            # 如果 use_context 为 True，则包括上下文输入
            if config.use_context:
                text_context_inputs = batch["text_context_tokens"].to(device)
                text_context_mask = batch["text_context_masks"].to(device)
                audio_context_inputs = batch["audio_context_inputs"].to(device)
                audio_context_mask = batch["audio_context_masks"].to(device)

            # 通过模型进行前向传播
            if config.use_context:
                outputs = model(
                    text_inputs, text_mask,
                    text_context_inputs, text_context_mask,
                    audio_inputs, audio_mask,
                    audio_context_inputs, audio_context_mask
                )
            else:
                outputs = model(
                    text_inputs, text_mask,
                    audio_inputs, audio_mask
                )

            # 提取多模态预测结果 ('M')
            preds = outputs['M'].cpu()  # 形状: [batch_size, 1]

            # 收集带有 video_id 和 clip_id 的预测结果
            for i in range(len(batch['video_id'])):
                predictions.append({
                    'video_id': batch['video_id'][i],  # 字符串
                    'clip_id': batch['clip_id'][i],    # 整数
                    'prediction': preds[i].item()      # 标量浮点值
                })

    # 更新现有 DataFrame 中的 'prediction' 列
    for pred in predictions:
        mask = (df['video_id'] == pred['video_id']) & (df['clip_id'] == pred['clip_id'])
        df.loc[mask, 'prediction'] = pred['prediction']

    # 将更新后的 DataFrame 保存回原 CSV 文件
    df.to_csv(csv_path, index=False)
    print(f"预测结果已更新至 {csv_path}")

if __name__ == "__main__":
    # 保存的模型权重路径（请更新为您的实际文件路径）
    model_weights_path = 'checkpoint/RH_acc_mosi_1_0.8734.pth'

    # 与训练设置匹配的配置
    config = EnConfig(
        batch_size=8,              # 根据训练时的情况调整
        dataset_name='test',       # 匹配您的测试数据集
        seed=1,                    # 与文件名中的种子匹配
        num_hidden_layers=5,       # run.py 中的默认值
        use_context=True,          # run.py 中的默认值
        use_attnFusion=True,       # run.py 中的默认值
        learning_rate=5e-6,
    )

    # 要更新的现有 CSV 文件路径
    csv_path = 'data/TEST/label.csv'

    # 生成预测并更新现有 CSV 文件
    generate_predictions(model_weights_path, config, csv_path)
