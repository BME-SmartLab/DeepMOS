# This is a modified version of the training script of https://github.com/nii-yamagishilab/mos-finetune-ssl

import os
import argparse
import fairseq
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import time
import json
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import random
import wandb
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np



# from> https://jonathanbgn.com/2021/08/30/audio-augmentation.html
class RandomClip:
    def __init__(self, clip_length=6000, sample_rate=8000):
        self.clip_length = clip_length
        self.vad = torchaudio.transforms.Vad(
            sample_rate=sample_rate, trigger_level=7.0)

    def __call__(self, audio_data):
        audio_length = audio_data.shape[1]
        if audio_length > self.clip_length:
            offset = random.randint(0, audio_length-self.clip_length)
            audio_data = audio_data[0,offset:(offset+self.clip_length)]

        return self.vad(audio_data) # remove silences at the beggining/end

class RandomSpeedChange:
    def __init__(self, sample_rate=8000):
        self.sample_rate = sample_rate

    def __call__(self, audio_data):
        speed_factor = random.choice([0.9, 1.0, 1.1])
        if speed_factor == 1.0: # no change
            return audio_data

        # change speed and resample to original rate:
        sox_effects = [
                    ["speed", str(speed_factor)],
                    ["rate", str(self.sample_rate)],
                    ]
        transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
                            audio_data, self.sample_rate, sox_effects)
        return transformed_audio


class ComposeTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio_data):
        for t in self.transforms:
            audio_data = t(audio_data)
        return audio_data



class MosPredictor(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim, nlstm, nunits, bidirectional, dropout_rate):
        super(MosPredictor, self).__init__()
        self.nunits=nunits 
        self.nlstm = nlstm
        nunits_fc=self.nunits
        self.bidirectional=bidirectional
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.dropout1 = nn.Dropout(dropout_rate/2)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(input_size=self.ssl_features,
                            hidden_size=self.nunits,
                            num_layers=self.nlstm,
                            batch_first=True,
                            bidirectional=self.bidirectional)

        if self.bidirectional:
            nunits_fc = 2*nunits_fc
        #nunits_fc = self.nlstm*nunits_fc
        self.output_layer1 = nn.Linear(nunits_fc, nunits_fc)
        self.output_layer2 = nn.Linear(nunits_fc, 1)
        self.activation = nn.SiLU()

    def forward(self, wav):
        #self.hidden = self.init_hidden(wav.shape[0])
        wav = wav.squeeze(1)  ## [batches, audio_len]
        res = self.ssl_model(wav, mask=False, features_only=True)

        x = res['x'] # [1,seq_length,embedding)
        batch_size, seq_len, _ = x.size()
        
        x = self.dropout1(x)
        
        x, _ = self.lstm(x)
        x = x[:,-1,:] # select last element from output
        
        x = self.dropout2(x)
        
        x = self.output_layer1(x)
        x = self.activation(x)
        
        x = self.dropout3(x)
        x = self.output_layer2(x)
        
        return x.squeeze(1)

    def init_hidden(self,batch_size):
        # the weights are of the form(nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.zeros((int(self.bidirectional)+1)*self.nlstm,batch_size,self.nunits)
        hidden_b = torch.zeros((int(self.bidirectional)+1)*self.nlstm,batch_size,self.nunits) #self.hparams.nb_lstm_layers, self.batch_size, self.nb_lstm_units)

        hidden_a = hidden_a.cuda()
        hidden_b = hidden_b.cuda()

        return (hidden_a, hidden_b)


def takeSecond(elem):
    a,b=elem
    return b
    
class MyDataset(Dataset):
    def __init__(self, wavdir, mos_list):
        self.mos_lookup = { }
        self.wavdir = wavdir
        f = open(mos_list, 'r')
        self.filesizes = []
        for line in f:
            parts = line.strip().split(',')
            wavname = parts[0]
            if len(parts)>1:
                mos = float(parts[1])
            else:
                mos = 1.0
            self.mos_lookup[wavname] = mos
            self.filesizes.append((wavname,self.get_file_size(wavname)))

        self.wavdir = wavdir
        #self.wavnames = sorted(self.mos_lookup.keys())
        self.wavnames= sorted(self.filesizes,key=takeSecond)
        
        self.audio_transforms = ComposeTransform([
                                    RandomSpeedChange(),
                                    RandomClip(),
                                    ])
    def get_file_size(self,wavname):
        wavpath = os.path.join(self.wavdir, wavname)
        return os.path.getsize(wavpath)
        
    def __getitem__(self, idx):
        wavname = self.wavnames[idx][0]
        wavpath = os.path.join(self.wavdir, wavname)
        wav = torchaudio.load(wavpath)[0]
        score = self.mos_lookup[wavname]
        
        return wav, score, wavname
    

    def __len__(self):
        return len(self.wavnames)


    def collate_fn(self, batch):  ## zero padding
        wavs, scores, wavnames = zip(*batch)
   
        wavs = list(wavs)
        max_len = max(wavs, key = lambda x : x.shape[1]).shape[1]
        output_wavs = []
        for wav in wavs:
            #padded_wav = self.audio_transforms(wav)
            amount_to_pad = max_len - wav.shape[1]
            padded_wav = torch.nn.functional.pad(wav, (0, amount_to_pad), 'constant', 0)
            output_wavs.append(padded_wav)

        output_wavs = torch.stack(output_wavs, dim=0)
        scores  = torch.stack([torch.tensor(x) for x in list(scores)], dim=0)
        return output_wavs, scores, wavnames

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
   
def main():

    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, help='Path of your DATA/ directory')
    parser.add_argument('--fairseq_base_model', type=str, required=True, help='Path to pretrained fairseq base model')
    parser.add_argument('--finetune_from_checkpoint', type=str, required=False, help='Path to your checkpoint to finetune from')
    parser.add_argument('--outdir', type=str, required=False, default='checkpoints', help='Output directory for your trained checkpoints')
    parser.add_argument('--nunits', type=int, required=False, default=16, help='Number of hidden units / lstm cells')
    parser.add_argument('--nlstm', type=int, required=False, default=1, help='Number of LSTM layers')
    parser.add_argument('--bidirectional', required=False, default=False, action='store_true', help='Bidirectional LSTM')
    parser.add_argument('--dropout_rate', type=float, required=False, default=0.0, help='Dropout rate')
    parser.add_argument('--lr', type=float, required=False, default=0.0001, help='Learning rate')
    parser.add_argument('--ngpus', type=int, required=False, default=1, help="Number of GPUs")
    parser.add_argument('--batch_size', type=int, required=False, default=1, help="Batch size (multiplied by number of GPUs)")
    parser.add_argument('--accumlation', type=int, required=False, default=1, help="Accumlation")
    parser.add_argument('--seed', type=int, required=False, default=1234, help='Random seed')
    parser.add_argument('--shuffle', type=str2bool, required=False, default=True, help='Shuffle')
    parser.add_argument('--wandb_project_name', type=str, required=False, default="", help='Wandb.ai project name')
    parser.add_argument('--wandb_entity_name', type=str, required=False, default="", help='Wandb.ai entity name')
    parser.add_argument('--freeze_SSL_model',type=str2bool, required=False, default=False, help='Freeze SSL model')
    
    args = parser.parse_args()
   
    if len(args.wandb_project_name)>0:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity_name)

    # Set Random seeds
    SEED=args.seed
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    if len(args.wandb_project_name)>0:
        wandb.config.update(args)    
    
    print(args)
    
    filestr = str(args.nlstm)+'-'+str(args.nunits)+'-'+str(args.bidirectional)+'-'+str(args.dropout_rate)+'-'+str(args.ngpus)+'-'+str(args.batch_size)+'-lrschedule-'+str(args.lr)
    cp_path = args.fairseq_base_model
    datadir = args.datadir
    ckptdir = args.outdir
    my_checkpoint = args.finetune_from_checkpoint

    acc_item=args.accumlation

    if not os.path.exists(ckptdir):
        os.system('mkdir -p ' + ckptdir)
    
    
    device = "cuda:0"
    print('DEVICE: ' + str(device))

    wavdir = os.path.join(datadir, 'wav')
    trainlist = os.path.join(datadir, 'sets/train_mos_list.txt')
    validlist = os.path.join(datadir, 'sets/val_mos_list.txt')

    ssl_model_type = cp_path.split('/')[-1]
    if ssl_model_type == 'wav2vec_small.pt':
        SSL_OUT_DIM = 768
    elif ssl_model_type in ['w2v_large_lv_fsh_swbd_cv.pt', 'xlsr_53_56k.pt', 'libri960_big.pt']:
        SSL_OUT_DIM = 1024
    else:
        print('*** ERROR *** SSL model type ' + ssl_model_type + ' not supported.')
        exit()

    print("Loading model",ssl_model_type)
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    print("Removing pretraining modules") 
    ssl_model.remove_pretraining_modules()
    if args.freeze_SSL_model:
        print("Freezing SSL model")
        for param in ssl_model.parameters():
            param.requires_grad = False
    trainset = MyDataset(wavdir, trainlist)#, transform=audio_transforms)
    trainloader = DataLoader(trainset, batch_size=args.ngpus*args.batch_size, shuffle=args.shuffle, num_workers=4, collate_fn=trainset.collate_fn)

    validset = MyDataset(wavdir, validlist)
    validloader = DataLoader(validset, batch_size=args.ngpus*args.batch_size, shuffle=False, num_workers=4, collate_fn=validset.collate_fn)

    net = MosPredictor(ssl_model, SSL_OUT_DIM, args.nlstm, args.nunits, args.bidirectional, args.dropout_rate)

    if my_checkpoint != None:  ## do (further) finetuning
        print("Loading pretrained model for finetuning....")
        state_dict= torch.load(my_checkpoint)

        # if the training was performed with DistributedParallel
        if ((list(state_dict.items())[0][0])[:7])=='module':
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove 'module.' of dataparallel
                new_state_dict[name]=v
        else:
            new_state_dict = state_dict

        net.load_state_dict(new_state_dict)
    
    net = net.to(device)
    if args.ngpus>1:
        net = nn.DataParallel(net, device_ids=range(args.ngpus)) 
    
    criterion = nn.L1Loss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=args.lr*acc_item, momentum=0.9)
    
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 0.00005*acc_item, 0.0005*acc_item, step_size_up=1+int(1000/acc_item), mode='triangular')

    PREV_VAL_LOSS=9999999999
    orig_patience=50
  
    patience=orig_patience
    training_start_time = time.time()
    print("Training....")
    
    for epoch in range(1,10001):
        STEPS=0
        net.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        optimizer.zero_grad()
        optimizer_run=False
        loss_acc=0
        for i, data in enumerate(trainloader, 0):
            inputs, labels, filenames = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.set_grad_enabled(True):
                outputs = net(inputs)
                loss = criterion(outputs, labels)/acc_item
            
                if args.ngpus>1:
                    loss=loss.mean()
                loss.backward()
                running_loss += loss.item()
                loss_acc+=loss.item()
                optimizer_run=False
                if (i+1)%acc_item==0:
                    optimizer.step()
                    STEPS += 1
                    
                    scheduler.step()
                    optimizer.zero_grad()
                    optimizer_run=True
                    print("{} step: loss={}".format(STEPS,loss_acc))
                    loss_acc=0
            
                    for device_i in range(args.ngpus):
                            device_ = "cuda:"+str(device_i)
                            with torch.cuda.device(device_):
                                torch.cuda.empty_cache()
        if not optimizer_run:
                optimizer.step()
                STEPS += 1
                scheduler.step()
        
        print('EPOCH: ' + str(epoch),'\t\tAVG EPOCH TRAIN LOSS: ' + str(running_loss / STEPS),"took {:.2f}s".format(time.time() - epoch_start_time))
        epoch_val_loss = 0.0
        net.eval()
        ## clear memory to avoid OOM
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        ## validation
        VALSTEPS=0
        outputs_label = []
        outputs_predictions = []
        for i, data in enumerate(validloader, 0):
            VALSTEPS+=1
            inputs, labels, filenames = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = net(inputs)
            output_quantized = (((outputs*8).type(torch.int)).type(torch.float)/8).cpu().detach().numpy().tolist()
            outputs_predictions += output_quantized
            outputs_label +=labels.cpu().detach().numpy().tolist()
            loss = criterion(outputs, labels)

            epoch_val_loss += loss.item()
            if i%20==0:
                for device_i in range(args.ngpus):
                    device_ = "cuda:"+str(device_i)
                    with torch.cuda.device(device_):
                        torch.cuda.empty_cache()
 
        avg_val_loss=epoch_val_loss/VALSTEPS    
        print('\t\t\tVALIDATION LOSS: ' + str(avg_val_loss))
        reg_plot_val = sns.regplot(y=outputs_predictions, x=outputs_label)

        if len(args.wandb_project_name)>0:
            wandb.log({"loss": running_loss/STEPS,
                       "validation loss": avg_val_loss,
                       "validation regplot": reg_plot_val.figure})
        if avg_val_loss < PREV_VAL_LOSS:
            PREV_VAL_LOSS=avg_val_loss
            fname =  ssl_model_type[:-3]+"_ckpt_{:.3f}_".format(avg_val_loss) + '_' +str(epoch) + '_' + filestr
            PATH = os.path.join(ckptdir,fname)
            with open(PATH+'.json', 'w') as f:
                args_dict =args.__dict__ 
                args_dict['finetune_from_checkpoint'] = fname
                json.dump(args_dict , f, indent=2)

            torch.save(net.state_dict(), PATH)
            patience = orig_patience
            print('\t\t\t*** Loss has decreased, saving model to',PATH)
            if len(args.wandb_project_name)>0:
                wandb.run.summary["best_val_loss"] = avg_val_loss
        else:
            patience-=1
            if patience == 0:
                print('loss has not decreased for ' + str(orig_patience) + ' epochs; early stopping at epoch ' + str(epoch))
                break
        
    print('Finished Training')
    print("took {:.2f}s".format(time.time() - training_start_time))

if __name__ == '__main__':
    main()

