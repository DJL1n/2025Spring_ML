# utils/data_loader.py
import torch
from torch import nn
import transformers
import torchaudio
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from torchvision import transforms
import cv2

class Dataset_mosi(torch.utils.data.Dataset):
    def __init__(self, csv_path, audio_directory, mode, config):
        df = pd.read_csv(csv_path)
        invalid_files = ['3aIQUQgawaI/12.wav', '94ULum9MYX0/2.wav', 'mRnEJOLkhp8/24.wav', 'aE-X_QdDaqQ/3.wav', '94ULum9MYX0/11.wav', 'mRnEJOLkhp8/26.wav']
        for f in invalid_files:
            video_id = f.split('/')[0]
            clip_id = f.split('/')[1].split('.')[0]
            df = df[~((df['video_id'] == video_id) & (df['clip_id'] == int(clip_id)))]

        df = df[df['mode'] == mode].sort_values(by=['video_id', 'clip_id']).reset_index()

        self.targets_M = df['label']
        # 文本
        df['text'] = df['text'].str[0] + df['text'].str[1::].apply(lambda x: x.lower())
        self.texts = df['text']
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        # 音频
        self.audio_file_paths = []
        for i in range(0, len(df)):
            file_name = str(df['video_id'][i]) + '/' + str(df['clip_id'][i]) + '.wav'
            file_path = audio_directory + "/" + file_name
            self.audio_file_paths.append(file_path)
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

        self.video_id = df['video_id']
        self.config = config
        self.text_context_length = self.config.text_context_length
        self.audio_context_length = self.config.audio_context_length

    def __getitem__(self, index):
        text = str(self.texts[index])

        text_context = ''
        for i in range(1, self.text_context_length + 1):
            if index - i < 0 or self.video_id[index] != self.video_id[index - i]:
                break
            else:
                context = str(self.texts[index - i])
                text_context = context + '</s>' + text_context

        tokenized_text = self.tokenizer(
            text,
            max_length=96,
            padding="max_length",  # Pad to the specified max_length.
            truncation=True,  # Truncate to the specified max_length.
            add_special_tokens=True,  # Whether to insert [CLS], [SEP], <s>, etc.
            return_attention_mask=True
        )

        # tokenize text context
        text_context = text_context[:-4]
        tokenized_context = self.tokenizer(
            text_context,
            max_length=96,
            padding="max_length",  # Pad to the specified max_length.
            truncation=True,  # Truncate to the specified max_length.
            add_special_tokens=True,  # Whether to insert [CLS], [SEP], <s>, etc.
            return_attention_mask=True
        )

        # load audio
        sound, _ = torchaudio.load(self.audio_file_paths[index])
        soundData = torch.mean(sound, dim=0, keepdim=False)

        # load audio context
        audio_context = torch.tensor([])
        for i in range(1, self.audio_context_length + 1):
            if index - i < 0 or self.video_id[index] != self.video_id[index - i]:
                break
            else:
                context, _ = torchaudio.load(self.audio_file_paths[index - i])
                contextData = torch.mean(context, dim=0, keepdim=False)
                audio_context = torch.cat((contextData, audio_context), 0)

        # extract audio features
        features = self.feature_extractor(soundData, sampling_rate=16000, max_length=96000, return_attention_mask=True,
                                          truncation=True, padding="max_length")
        audio_features = torch.tensor(np.array(features['input_values']), dtype=torch.float32).squeeze()
        audio_masks = torch.tensor(np.array(features['attention_mask']), dtype=torch.long).squeeze()

        # extract audio context features
        if len(audio_context) == 0:
            audio_context_features = torch.zeros(96000)
            audio_context_masks = torch.zeros(96000)
        else:
            features = self.feature_extractor(audio_context, sampling_rate=16000, max_length=96000,
                                              return_attention_mask=True, truncation=True, padding="max_length")
            audio_context_features = torch.tensor(np.array(features['input_values']), dtype=torch.float32).squeeze()
            audio_context_masks = torch.tensor(np.array(features['attention_mask']), dtype=torch.long).squeeze()

        return {  # text
            "text_tokens": torch.tensor(tokenized_text["input_ids"], dtype=torch.long),
            "text_masks": torch.tensor(tokenized_text["attention_mask"], dtype=torch.long),
            "text_context_tokens": torch.tensor(tokenized_context["input_ids"], dtype=torch.long),
            "text_context_masks": torch.tensor(tokenized_context["attention_mask"], dtype=torch.long),
            # audio
            "audio_inputs": audio_features,
            "audio_masks": audio_masks,
            "audio_context_inputs": audio_context_features,
            "audio_context_masks": audio_context_masks,
            # labels
            "targets": torch.tensor(self.targets_M[index], dtype=torch.float),
            "video_id": self.video_id[index],
        }

    def __len__(self):
        return len(self.targets_M)

def data_loader(config):
    batch_size = config.batch_size
    dataset = config.dataset_name
    if dataset == 'mosi':
        csv_path = 'data/MOSI/label.csv'
        audio_file_path = "data/MOSI/wav"
        train_data = Dataset_mosi(csv_path, audio_file_path, 'train', config)
        test_data = Dataset_mosi(csv_path, audio_file_path, 'test', config)
        val_data = Dataset_mosi(csv_path, audio_file_path, 'valid', config)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, val_loader
    elif dataset == 'mosei':
        csv_path = 'data/MOSEI/label.csv'
        audio_file_path = "data/MOSEI/wav"
        train_data = Dataset_mosi(csv_path, audio_file_path, 'train', config)
        test_data = Dataset_mosi(csv_path, audio_file_path, 'test', config)
        val_data = Dataset_mosi(csv_path, audio_file_path, 'valid', config)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, val_loader
    elif dataset == 'test':
        csv_path = 'data/test/label.csv'
        audio_file_path = "data/test/wav"
        test_data = Dataset_mosi(csv_path, audio_file_path, 'test', config)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        return test_loader
