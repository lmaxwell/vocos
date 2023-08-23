
import os 

os.environ["MKL_NUM_THREADS"]="2"
os.environ["OMP_NUM_THREADS"]="2"
import torchaudio
import sys
import torch
import time


from vocos import Vocos

def load(config_path,model_path):
    model=Vocos.from_hparams(config_path)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict['state_dict'],strict=False)
    model.eval()
    return model


config_path,model_path,audio_path,output_path = sys.argv[1:]
vocos = load(config_path,model_path)

mel = torch.randn(1, 100, 256)  # B, C, T
audio = vocos.decode(mel)




y, sr = torchaudio.load(audio_path)
if y.size(0) > 1:  # mix to mono
    y = y.mean(dim=0, keepdim=True)
y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=24000)
y_hat = vocos(y)


mel = vocos.feature_extractor(y)

vocos.backbone.init_state(1)

step = 4
i=0
stft = []

start_time = time.time()
while i+step < mel.shape[2] - 1:
    print(i,i+step,mel.shape[2])
    stft_ = vocos.backbone.stream(mel[:,:,i:i+step])
    stft.append(stft_)
    i += step
stft_ = vocos.backbone.stream(mel[:,:,i:])
stft.append(stft_)
stft = torch.cat(stft,dim=1)

audio = vocos.head(stft)
print("cost {}".format(time.time()-start_time))
torchaudio.save(output_path+"_stream.wav",audio,sr)


    

torchaudio.save(output_path,y_hat,sr)


