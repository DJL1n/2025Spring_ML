# utils/train.py
import torch
from torch import nn
from tqdm import tqdm
from utils.metricsTop import MetricsTop
from utils.model import Model
from utils.baseline import baseline
import random
import numpy as np
from utils.data_loader import data_loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " % (key, src_dict[key])
    return dst_str

class Trainer:
    def __init__(self, config):
        self.config = config
        self.criterion = nn.L1Loss() if config.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(config.train_mode).getMetics(config.dataset_name)
        self.tasks = config.tasks
        self.current_epoch = 0

    def do_train(self, model, data_loader):
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)

        accumulation_steps = self.config.gradient_accumulation_steps

        total_loss = 0.0
        optimizer.zero_grad()  # 在训练循环开始前清零梯度

        for i, batch in enumerate(tqdm(data_loader)):
            text_inputs = batch["text_tokens"].to(device)
            text_mask = batch["text_masks"].to(device)
            text_context_inputs = batch["text_context_tokens"].to(device)
            text_context_mask = batch["text_context_masks"].to(device)

            audio_inputs = batch["audio_inputs"].to(device)
            audio_mask = batch["audio_masks"].to(device)
            audio_context_inputs = batch["audio_context_inputs"].to(device)
            audio_context_mask = batch["audio_context_masks"].to(device)
            targets = batch["targets"].to(device).view(-1, 1)

            optimizer.zero_grad()

            if self.config.use_context:
                outputs = model(text_inputs, text_mask, text_context_inputs, text_context_mask ,
                                audio_inputs, audio_mask, audio_context_inputs, audio_context_mask,)
            else:
                outputs = model(text_inputs, text_mask, audio_inputs, audio_mask)

            loss = 0.0
            for m in self.tasks:
                sub_loss = self.config.loss_weights[m] * self.criterion(outputs[m], targets)
                loss += sub_loss
            # 缩放损失以保持梯度幅度一致
            total_loss_batch = total_loss_batch / accumulation_steps

            # 反向传播，累积梯度
            total_loss_batch.backward()

            # 累积总损失（还原缩放效果）
            total_loss += total_loss_batch.item() * accumulation_steps * text_inputs.size(0)

            # 当达到累积步数时，更新参数并清零梯度
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # 处理训练循环末尾的剩余梯度
        if len(data_loader) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss = round(total_loss / len(data_loader.dataset), 4)
        return total_loss

    def do_test(self, model, data_loader, mode):
        model.eval()

        y_pred = {'M': [], 'T': [], 'A': []}
        y_true = {'M': [], 'T': [], 'A': []}
        total_loss = 0
        total_cf_loss = 0
        val_loss = {'M': 0, 'T': 0, 'A': 0}


        with torch.no_grad():
            for batch in tqdm(data_loader):
                text_inputs = batch["text_tokens"].to(device)
                text_mask = batch["text_masks"].to(device)
                text_context_inputs = batch["text_context_tokens"].to(device)
                text_context_mask = batch["text_context_masks"].to(device)

                audio_inputs = batch["audio_inputs"].to(device)
                audio_mask = batch["audio_masks"].to(device)
                audio_context_inputs = batch["audio_context_inputs"].to(device)
                audio_context_mask = batch["audio_context_masks"].to(device)

                targets = batch["targets"].to(device).view(-1, 1)

                if self.config.use_context:
                    outputs = model(text_inputs, text_mask, text_context_inputs, text_context_mask, audio_inputs, audio_mask, audio_context_inputs, audio_context_mask)
                else:
                    outputs = model(text_inputs, text_mask, audio_inputs, audio_mask)

                loss = 0.0
                for m in self.tasks:
                    sub_loss = self.config.loss_weights[m] * self.criterion(outputs[m], targets)
                    # print(m, outputs[m],targets)
                    loss += sub_loss
                    val_loss[m] += sub_loss.item() * text_inputs.size(0)
                total_loss += loss.item() * text_inputs.size(0)
                for m in self.tasks:
                    y_pred[m].append(outputs[m].cpu())
                    y_true[m].append(targets.cpu())

        for m in self.tasks:
            val_loss[m] = round(val_loss[m] / len(data_loader.dataset), 4)
        total_loss = round(total_loss / len(data_loader.dataset), 4)
        print(mode + " >> loss: ", total_loss, "   M_loss: ", val_loss['M'], "  T_loss: ", val_loss['T'], "  A_loss: ", val_loss['A'])

        eval_results = {}
        for m in self.tasks:
            pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
            results = self.metrics(pred, true)
            print('%s: >> ' % (m) + dict_to_str(results))
            eval_results[m] = results
        eval_results = eval_results[self.tasks[0]]
        eval_results['Loss'] = total_loss
        return eval_results

def EnRun(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True

    train_loader, test_loader, val_loader = data_loader(config)
    if not config.use_context and not config.use_counterfactual:
        model = baseline(config).to(device)
    else:
        model = Model(config).to(device)
    # for param in model.data2vec_model.feature_extractor.parameters():
    #     param.requires_grad = False

    trainer = Trainer(config,device)

    lowest_eval_loss = 100
    highest_eval_acc = 0
    epoch = 0
    best_epoch = 0
    prev_loss_model_path = None  # 记录之前基于损失的模型路径
    prev_acc_model_path = None  # 记录之前基于准确率的模型路径

    while True:
        print('---------------------EPOCH: ', epoch, '--------------------')
        epoch += 1
        trainer.do_train(model, train_loader)
        eval_results = trainer.do_test(model, val_loader, "VAL")

        if eval_results['Loss'] <= lowest_eval_loss:
            # 如果存在之前的基于损失的模型文件，则删除
            if prev_loss_model_path and os.path.exists(prev_loss_model_path):
                os.remove(prev_loss_model_path)

            # 更新最小损失并保存新模型
            lowest_eval_loss = eval_results['Loss']
            model_path = config.model_save_path + f'RH_loss_{config.dataset_name}_{config.seed}_{lowest_eval_loss:.4f}.pth'
            torch.save(model.state_dict(), model_path)
            prev_loss_model_path = model_path  # 更新路径
            best_epoch = epoch  # 更新最佳 epoch

        if eval_results['Has0_acc_2'] >= highest_eval_acc:
            # 如果存在之前的基于准确率的模型文件，则删除
            if prev_acc_model_path and os.path.exists(prev_acc_model_path):
                os.remove(prev_acc_model_path)

            # 更新最高准确率并保存新模型
            highest_eval_acc = eval_results['Has0_acc_2']
            model_path = config.model_save_path + f'RH_acc_{config.dataset_name}_{config.seed}_{highest_eval_acc:.4f}.pth'
            torch.save(model.state_dict(), model_path)
            prev_acc_model_path = model_path  # 更新路径
            torch.save(model.state_dict(), config.model_save_path + f'RH_acc_{config.dataset_name}_{config.seed}_{highest_eval_acc}.pth')
        if epoch - best_epoch >= config.early_stop:
            break

    model.load_state_dict(torch.load(config.model_save_path + f'RH_acc_{config.dataset_name}_{config.seed}_{highest_eval_acc}.pth'))
    test_results_loss = trainer.do_test(model, test_loader, "TEST")
    print('%s: >> ' % ('TEST (highest val acc) ') + dict_to_str(test_results_loss))

    model.load_state_dict(torch.load(config.model_save_path + f'RH_loss_{config.dataset_name}_{config.seed}_{lowest_eval_loss}.pth'))
    test_results_acc = trainer.do_test(model, test_loader, "TEST")
    print('%s: >> ' % ('TEST (lowest val loss) ') + dict_to_str(test_results_acc))
