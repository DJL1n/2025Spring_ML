import torch
import pandas as pd
from utils.model import Model
from utils.config import EnConfig
from utils.data_loader import data_loader
import os

def generate_predictions(model_weights_path, config, csv_path):
    """
    加载保存的模型权重，在测试数据集上生成预测，并更新现有 CSV 文件中的 'prediction' 列。
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loader = data_loader(config)
    model = Model(config).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    df = pd.read_csv(csv_path)
    print("CSV 中的列:", df.columns.tolist())
    print("CSV 前几行:", df.head())

    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            text_inputs = batch["text_tokens"].to(device)
            text_mask = batch["text_masks"].to(device)
            audio_inputs = batch["audio_inputs"].to(device)
            audio_mask = batch["audio_masks"].to(device)

            if config.use_context:
                text_context_inputs = batch["text_context_tokens"].to(device)
                text_context_mask = batch["text_context_masks"].to(device)
                audio_context_inputs = batch["audio_context_inputs"].to(device)
                audio_context_mask = batch["audio_context_masks"].to(device)
                outputs = model(
                    text_inputs, text_mask,
                    text_context_inputs, text_context_mask,
                    audio_inputs, audio_mask,
                    audio_context_inputs, audio_context_mask
                )
            else:
                outputs = model(text_inputs, text_mask, audio_inputs, audio_mask)

            preds = outputs['M'].cpu()

            for i in range(len(batch['video_id'])):
                # 将 clip_id 从张量转换为整数
                clip_id = batch['clip_id'][i].item() if torch.is_tensor(batch['clip_id'][i]) else batch['clip_id'][i]
                predictions.append({
                    'video_id': batch['video_id'][i],
                    'clip_id': clip_id,
                    'prediction': preds[i].item()
                })

    print("预测数量:", len(predictions))
    if predictions:
        print("示例预测:", predictions[:2])

    for pred in predictions:
        mask = (df['video_id'].astype(str) == str(pred['video_id'])) & (df['clip_id'].astype(int) == pred['clip_id'])
        if mask.sum() == 0:
            print(f"未找到匹配: video_id={pred['video_id']}, clip_id={pred['clip_id']}")
        else:
            df.loc[mask, 'prediction'] = pred['prediction']
            print(f"已更新: video_id={pred['video_id']}, clip_id={pred['clip_id']}, prediction={pred['prediction']}")

    print("更新后 DataFrame 前几行:", df.head())
    df.to_csv(csv_path, index=False)
    print(f"预测结果已更新至 {csv_path}")

if __name__ == "__main__":
    model_weights_path = 'checkpoint/RH_acc_mosi_1_0.8734.pth'
    config = EnConfig(
        batch_size=8,
        dataset_name='test',
        seed=1,
        num_hidden_layers=5,
        use_context=True,
        use_attnFusion=True,
        learning_rate=5e-6,
    )
    csv_path = 'data/TEST/label.csv'
    generate_predictions(model_weights_path, config, csv_path)
