import os
import torch
import torchaudio
import cv2
import pickle
import numpy as np
from facenet_pytorch import MTCNN
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from transformers import AutoProcessor

class MultimodalDataset(Dataset):
    def __init__(self, folder_path, audio_target_length=16000*3, video_frames_count=30, audio_n_mels=128, audio_n_fft=1024, audio_hop_length=512):
        self.folder_path = folder_path
        self.audio_target_length = audio_target_length
        self.video_frames_count = video_frames_count
        self.audio_n_mels = audio_n_mels
        self.audio_n_fft = audio_n_fft
        self.audio_hop_length = audio_hop_length

        self.audio_processor = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_mels=audio_n_mels,
            n_fft=audio_n_fft,
            hop_length=audio_hop_length,
            power=2.0
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.mtcnn = MTCNN(keep_all=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.data = self._process_folder()

    def _process_folder(self):
        data = []
        for video_file in tqdm(os.listdir(self.folder_path)):
            if video_file.endswith(".mp4"):
                parts = video_file.split("-")
                modality = int(parts[0])
                statement_id = int(parts[4])
                emotion = int(parts[2]) - 1
                
                if modality != 1:
                    continue

                video_path = os.path.join(self.folder_path, video_file)
                text = "Kids are talking by the door" if statement_id == 1 else "Dogs are sitting by the door"

                audio_embedding = self._process_audio(video_path)
                video_frames = self._process_video(video_path)

                if audio_embedding is not None and video_frames is not None:
                    data.append({
                        "audio": audio_embedding,
                        "video": video_frames,
                        "text": text,
                        "emotion": emotion
                    })
        return data

    def _process_audio(self, video_path):
        try:
            waveform, sample_rate = torchaudio.load(video_path, format="mp4")
            waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-7)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            if waveform.shape[1] < self.audio_target_length:
                pad_amount = self.audio_target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

            elif waveform.shape[1] > self.audio_target_length:
                start = torch.randint(0, waveform.shape[1] - self.audio_target_length, (1,)).item()
                waveform = waveform[:, start:start+self.audio_target_length]

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            mel_spec = self.audio_processor(waveform)
            log_mel_spec = self.amplitude_to_db(mel_spec)
            log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
            
            resized_spec = torch.nn.functional.interpolate(
                log_mel_spec.unsqueeze(0), 
                size=(112, 112), 
                mode="bilinear", 
                align_corners=False
            ).squeeze(0)
            
            # spectrogram_img = resized_spec.repeat(3, 1, 1)
            # print("Audio shape:", spectrogram_img.shape)
            return resized_spec.numpy()
        
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None

    def _process_video(self, video_path):
        try:
            vidcap = cv2.VideoCapture(video_path)
            success, image = vidcap.read()
            frames = []
            count = 0

            while success and len(frames) < self.video_frames_count:
                if count % 10 == 0:  # Take every 10th frame
                    boxes, _ = self.mtcnn.detect(image)
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box)
                            detected_face = image[y1:y2, x1:x2]
                            if detected_face.size != 0:
                                detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
                                detected_face = cv2.resize(detected_face, (112, 112))
                                detected_face = cv2.normalize(
                                    detected_face, None, alpha=0, 
                                    beta=1, norm_type=cv2.NORM_MINMAX, 
                                    dtype=cv2.CV_32F
                                )
                                frames.append(detected_face)
                                break
                success, image = vidcap.read()
                count += 1

            if len(frames) != 0:
                return np.stack(frames[:self.video_frames_count])
            else:
                return None
            
        except Exception as e:
            print(f"Error processing video: {e}")
            return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



class MultimodalDataProcessor:
    def __init__(self, pickle_file, batch_size=32, test_size=0.3, val_size=0.5, random_state=42):
        self.pickle_file = pickle_file
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.class_weights = None

    def load_data(self):
        with open(self.pickle_file, 'rb') as f:
            self.data = pickle.load(f)

    def split_data(self):
        # Ensure uniform distribution of the 'emotion' labels
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)

        # Split into train and temp sets
        for train_idx, temp_idx in splitter.split(self.data, [item['emotion'] for item in self.data]):
            self.train_data = [self.data[i] for i in train_idx]
            temp_data = [self.data[i] for i in temp_idx]

        # Further split temp into validation and test sets
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.val_size, random_state=self.random_state)
        for val_idx, test_idx in splitter.split(temp_data, [item['emotion'] for item in temp_data]):
            self.val_data = [temp_data[i] for i in val_idx]
            self.test_data = [temp_data[i] for i in test_idx]

    def compute_class_weights(self):
        # Compute class weights for cross-entropy loss
        emotion_counts = Counter([item['emotion'] for item in self.train_data])
        total_samples = len(self.train_data)
        self.class_weights = torch.tensor(
            [total_samples / (len(emotion_counts) * emotion_counts[i]) for i in range(len(emotion_counts))],
            dtype=torch.float32
        )

    class MultimodalDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            audio = F.interpolate(torch.tensor(item['audio'], dtype=torch.float32).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
            audio = audio.repeat(3, 1, 1)  # Repeat the audio into 3 channels

            # Ensure the video has exactly 12 frames
            if item['video'].shape[0] >= 12:
                video_frames = item['video'][:12]
            else:
                video_frames = np.concatenate(
                    [item['video'], np.repeat(item['video'][-1][np.newaxis, ...], 12 - item['video'].shape[0], axis=0)],
                    axis=0
                )
            video = F.interpolate(torch.tensor(video_frames, dtype=torch.float32).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
            text = item['text']  # Assuming text is a string
            emotion = torch.tensor(item['emotion'], dtype=torch.long)
            return {'audio': audio, 'video': video, 'text': text, 'emotion': emotion}

    def get_dataloader(self, data, shuffle=True):
        dataset = self.MultimodalDataset(data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataloader

    def process(self):
        self.load_data()
        self.split_data()
        self.compute_class_weights()
        train_dataloader = self.get_dataloader(self.train_data, shuffle=True)
        val_dataloader = self.get_dataloader(self.val_data, shuffle=False)
        test_dataloader = self.get_dataloader(self.test_data, shuffle=False)
        return train_dataloader, val_dataloader, test_dataloader, self.class_weights




# if __name__ == "__main__":
#     dataset_folder = r"ravdess_all_videos"
#     multimodal_dataset = MultimodalDataset(dataset_folder)

#     with open("multimodal_dataset_temp.pickle", "wb") as f:
#         pickle.dump(multimodal_dataset.data, f, protocol=pickle.HIGHEST_PROTOCOL)