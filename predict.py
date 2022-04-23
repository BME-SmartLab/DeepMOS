# This is a modified version of the prediction script of https://github.com/nii-yamagishilab/mos-finetune-ssl

import os
import argparse
import torch
import torch.nn as nn
import fairseq
from torch.utils.data import DataLoader
from deepmos import MosPredictor, MyDataset
import numpy as np
import scipy.stats
import json

def systemID(uttID):
    return uttID.split('-')[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, required=True, help='The JSON file that contains the settings and pretrained model.')
    parser.add_argument('--datadir', type=str, required=True, help='Path of your DATA/ directory')
    parser.add_argument('--outfile', type=str, required=False, default='answer.txt', help='Output filename for your answer.txt file for submission to the CodaLab leaderboard.')
 
    args = parser.parse_args()
    datadir = args.datadir
    outfile = args.outfile


    with open(args.json, 'r') as f:
        args.__dict__ = json.load(f)

    print(args)
    cp_path = args.fairseq_base_model
    my_checkpoint = "checkpoints/"+args.finetune_from_checkpoint #finetuned_checkpoint
    system_csv_path = os.path.join(datadir, 'mydata_system.csv')

    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    ssl_model = model[0]
    ssl_model.remove_pretraining_modules()

    print('Loading checkpoint')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ssl_model_type = cp_path.split('/')[-1]
    if ssl_model_type == 'wav2vec_small.pt':
        SSL_OUT_DIM = 768
    elif ssl_model_type in ['w2v_large_lv_fsh_swbd_cv.pt', 'xlsr_53_56k.pt','libri960_big.pt']:
        SSL_OUT_DIM = 1024
    else:
        print('*** ERROR *** SSL model type ' + ssl_model_type + ' not supported.')
        exit()

    model = MosPredictor(ssl_model, SSL_OUT_DIM, args.nlstm, args.nunits, args.bidirectional, args.dropout_rate)

    #model = MosPredictor(ssl_model, SSL_OUT_DIM, args.nunits, args.bidirectional, args.dropout_rate, args.nlstm)
    #model = MosPredictor(ssl_model, SSL_OUT_DIM).to(device)

    model.eval()

    state_dict= torch.load(my_checkpoint)
    if ((list(state_dict.items())[0][0])[:7])=='module': # the training was performed with DistributedParallel
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.' of dataparallel
            new_state_dict[name]=v
    else:
        new_state_dict = state_dict

    model.load_state_dict(new_state_dict)

    wavdir = os.path.join(datadir, 'wav')
    validlist = os.path.join(datadir, 'sets/val_mos_list.txt')
    model = model.to(device)
    print('Loading data')
    validset = MyDataset(wavdir, validlist)
    validloader = DataLoader(validset, batch_size=1, shuffle=True, num_workers=1, collate_fn=validset.collate_fn)

    total_loss = 0.0
    num_steps = 0.0
    predictions = { }  # filename : prediction
    criterion = nn.L1Loss()
    print('Starting prediction')

    for i, data in enumerate(validloader, 0):
        inputs, labels, filenames = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        if inputs.shape[-1]>500000:
            print("Input data truncated from",inputs.shape[-1]) 
            inputs = inputs[:,:,:500000]
        print(i,inputs.shape)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        output = outputs.cpu().detach().numpy()[0]
        predictions[filenames[0]] = output  ## batch size = 1
        
        if i%1==0:
            for device_i in range(args.ngpus):
                device_ = "cuda:"+str(device_i)
                with torch.cuda.device(device_):
                    torch.cuda.empty_cache()

    true_MOS = { }
    validf = open(validlist, 'r')
    for line in validf:
        parts = line.strip().split(',')
        uttID = parts[0]
        MOS = float(parts[1])
        true_MOS[uttID] = MOS

    ## compute correls.
    sorted_uttIDs = sorted(predictions.keys())
    ts = []
    ps = []
    for uttID in sorted_uttIDs:
        t = true_MOS[uttID]
        p = predictions[uttID]
        ts.append(t)
        ps.append(p)

    truths = np.array(ts)
    preds = np.array(ps)

    ### UTTERANCE
    MSE=np.mean((truths-preds)**2)
    print('[UTTERANCE] Test error= %f' % MSE)
    LCC=np.corrcoef(truths, preds)
    print('[UTTERANCE] Linear correlation coefficient= %f' % LCC[0][1])
    SRCC=scipy.stats.spearmanr(truths.T, preds.T)
    print('[UTTERANCE] Spearman rank correlation coefficient= %f' % SRCC[0])
    KTAU=scipy.stats.kendalltau(truths, preds)
    print('[UTTERANCE] Kendall Tau rank correlation coefficient= %f' % KTAU[0])

    ### SYSTEM
    true_sys_MOS_avg = { }
    csv_file = open(system_csv_path, 'r')
    csv_file.readline()  ## skip header
    for line in csv_file:
        parts = line.strip().split(',')
        sysID = parts[0]
        MOS = float(parts[1])
        true_sys_MOS_avg[sysID] = MOS

    pred_sys_MOSes = { }
    for uttID in sorted_uttIDs:
        sysID = systemID(uttID)
        noop = pred_sys_MOSes.setdefault(sysID, [ ])
        pred_sys_MOSes[sysID].append(predictions[uttID])

    pred_sys_MOS_avg = { }
    for k, v in pred_sys_MOSes.items():
        avg_MOS = sum(v) / (len(v) * 1.0)
        pred_sys_MOS_avg[k] = avg_MOS

    ## make lists sorted by system
    pred_sysIDs = sorted(pred_sys_MOS_avg.keys())
    sys_p = [ ]
    sys_t = [ ]
    for sysID in pred_sysIDs:
        sys_p.append(pred_sys_MOS_avg[sysID])
        sys_t.append(true_sys_MOS_avg[sysID])

    sys_true = np.array(sys_t)
    sys_predicted = np.array(sys_p)

    MSE=np.mean((sys_true-sys_predicted)**2)
    print('[SYSTEM] Test error= %f' % MSE)
    LCC=np.corrcoef(sys_true, sys_predicted)
    print('[SYSTEM] Linear correlation coefficient= %f' % LCC[0][1])
    SRCC=scipy.stats.spearmanr(sys_true.T, sys_predicted.T)
    print('[SYSTEM] Spearman rank correlation coefficient= %f' % SRCC[0])
    KTAU=scipy.stats.kendalltau(sys_true, sys_predicted)
    print('[SYSTEM] Kendall Tau rank correlation coefficient= %f' % KTAU[0])

    ## generate answer.txt for codalab
    ans = open(outfile, 'w')
    for k, v in predictions.items():
        outl = k.split('.')[0] + ',' + str(v) + '\n'
        ans.write(outl)
    ans.close()

if __name__ == '__main__':
    main()
