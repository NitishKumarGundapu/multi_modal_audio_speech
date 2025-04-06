import torch as t
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import moviepy
import shutil
import subprocess
import argparse
import glob
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import VideoFileClip

import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from ipdb import set_trace
import pickle as pkl
import h5py
import soundfile as sf
import torchaudio
import torchvision
import glob


### VGGSound
from scipy import signal
import soundfile as sf
###

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def get_raw_audio_files(video_pth,save_pth):
    
    def get_audio_wav(name, spth, audio_name):
        video = VideoFileClip(name)
        audio = video.audio
        audio.write_audiofile(os.path.join(spth, audio_name), fps=16000)

    sound_list = os.listdir(video_pth)
    for audio_id in sound_list:
        name = os.path.join(video_pth, audio_id)
        audio_name = audio_id[:-4] + '.wav'
        exist_lis = os.listdir(save_pth)
        if audio_name in exist_lis:
            print("already exist!")
            continue
        try:
            get_audio_wav(name, save_pth, audio_name)
            print("finish video id: " + audio_name)
        except:
            print("cannot load ", name)

    print("\n------------------------------ end -------------------------------\n")
    
    

class AVE_dataset(Dataset):

	def __init__(self,audio_dir,video_dir,vis_encoder_type = "vit"):
		data = pd.read_csv("/raid/amana/lavish_multi_model/emotion_detection/data/text_data.csv")
		data['filename'] = [f'dia{a}_utt{b}' for a,b in zip(data['Dialogue_ID'],data['Utterance_ID'])]
		self.labels = data['Emotion']
		self.num_classes = len(set(self.labels))
		self.audio_length = 1
		self.audio_folder = audio_dir
		self.video_folder = video_dir
		self.vis_encoder_type = vis_encoder_type
		self.raw_gt = data['filename']

		if self.vis_encoder_type == 'vit':
			self.norm_mean = -4.1426
			self.norm_std = 3.2001
		
		elif self.vis_encoder_type == 'swin':
			self.norm_mean =  -4.984795570373535
			self.norm_std =  3.7079780101776123
			

		if self.vis_encoder_type == 'vit':
			self.my_normalize = Compose([
				Resize([224,224], interpolation=Image.BICUBIC),
				Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
			])
		elif self.vis_encoder_type == 'swin':
			self.my_normalize = Compose([
				Resize([192,192], interpolation=Image.BICUBIC),
				Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
			])

   
	def getVggoud_proc(self, filename, idx=None):
		audio_length = 1
		samples, samplerate = sf.read(filename)

		if samples.shape[0] > 16000*(audio_length+0.1):
			sample_indx = np.linspace(0, samples.shape[0]-16000*(self.audio_length+0.1), num=10, dtype=int)
			samples = samples[sample_indx[idx]:sample_indx[idx]+int(16000*self.audio_length)]
		else:
			samples = np.tile(samples,int(self.audio_length))[:int(16000*self.audio_length)]

		samples[samples > 1.] = 1.
		samples[samples < -1.] = -1.

		_,__, spectrogram = signal.spectrogram(samples, samplerate)
		spectrogram = np.log(spectrogram+ 1e-7)
		mean = np.mean(spectrogram)
		std = np.std(spectrogram)
		spectrogram = np.divide(spectrogram-mean,std+1e-9)
  
		return torch.tensor(spectrogram).float()


	def _wav2fbank(self, filename, filename2=None, idx=None):
		# mixup
		if filename2 == None:
			waveform, sr = torchaudio.load(filename)
			waveform = waveform - waveform.mean()
		# mixup
		else:
			waveform1, sr = torchaudio.load(filename)
			waveform2, _ = torchaudio.load(filename2)

			waveform1 = waveform1 - waveform1.mean()
			waveform2 = waveform2 - waveform2.mean()

			if waveform1.shape[1] != waveform2.shape[1]:
				if waveform1.shape[1] > waveform2.shape[1]:
					temp_wav = torch.zeros(1, waveform1.shape[1])
					temp_wav[0, 0:waveform2.shape[1]] = waveform2
					waveform2 = temp_wav
				else:
					waveform2 = waveform2[0, 0:waveform1.shape[1]]

			mix_lambda = np.random.beta(10, 10)
			mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
			waveform = mix_waveform - mix_waveform.mean()
		
		if waveform.shape[1] > 16000*(self.audio_length+0.1):
			sample_indx = np.linspace(0, waveform.shape[1] -16000*(self.audio_length+0.1), num=10, dtype=int)
			waveform = waveform[:,sample_indx[idx]:sample_indx[idx]+int(16000*self.audio_length)]


		if self.vis_encoder_type == 'vit':
			fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
		elif self.vis_encoder_type == 'swin':
			fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=192, dither=0.0, frame_shift=5.2)

		########### ------> very important: audio normalized
		fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
		### <--------
		if self.vis_encoder_type == 'vit':
			target_length = int(1024 * (1/10)) ## for audioset: 10s
		elif self.vis_encoder_type == 'swin':
			target_length = 192 ## yb: overwrite for swin

		n_frames = fbank.shape[0]
		p = target_length - n_frames

		# cut and pad
		if p > 0:
			m = torch.nn.ZeroPad2d((0, 0, 0, p))
			fbank = m(fbank)
		elif p < 0:
			fbank = fbank[0:target_length, :]

		if filename2 == None:
			return fbank, 0
		else:
			return fbank, mix_lambda

	def __len__(self):
		return len(self.labels)

	def _one_hot_encoding(self,x,num_classes):
		l = [0]*num_classes
		l[x-1] = 1
		return t.tensor(l)

	def __getitem__(self, idx):
		file_name = self.raw_gt.iloc[idx]
		total_audio = []
		for audio_sec in range(10):
			fbank, _ = self._wav2fbank(self.audio_folder+'/'+file_name+ '.wav', idx=audio_sec)
			total_audio.append(fbank)
		total_audio = torch.stack(total_audio)


		total_num_frames = len(glob.glob(self.video_folder+'/'+file_name+'/*.jpg'))
		sample_indx = np.linspace(1, total_num_frames , num=10, dtype=int)
		total_img = []
		for vis_idx in range(10):
			tmp_idx = sample_indx[vis_idx]
			tmp_img = torchvision.io.read_image(self.video_folder+'/'+file_name+'/'+ str("{:08d}".format(tmp_idx))+ '.jpg')/255
			tmp_img = self.my_normalize(tmp_img)
			total_img.append(tmp_img)
		total_img = torch.stack(total_img)
  
		emotion_id = {'anger' : 0,
              'disgust' : 1, 'fear' : 3, 'joy':4, 
              'neutral': 5, 'sadness': 6, 'surprise':2}

		return {
                'audio_spec': total_audio, 
				'GT':self.labels[idx],
				'emotion_arr' : emotion_id[self.labels[idx]],
				'image':total_img,
            }    