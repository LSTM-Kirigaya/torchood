import argparse
import time
import os

import yaml
from sklearn.metrics import (precision_score,
                             recall_score,
                             f1_score,
                             classification_report)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from torch import optim
# from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm
import wandb
from sklearn.metrics import roc_auc_score

from evaluation import *
from isic2019 import ISIC

from torchood import *
import models

import warnings
warnings.filterwarnings('ignore')

def check_dir(test_dir):
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)


def valid_id(outputs_, labels_):
    pred = torch.argmax(outputs_, dim=1).detach().cpu().numpy()
    target = torch.argmax(labels_, dim=1).detach().cpu().numpy()
    result = {'pre': precision_score(target, pred, average='weighted'),
              'rec': recall_score(target, pred, average='weighted'),
              'f1s': f1_score(target, pred, average='weighted')}
    return result


def exp_rampup(epoch, warmup):
    if warmup == 0:
        return 1.0
    else:
        current = np.clip(epoch, 0, warmup)
        phase = 1.0 - current / warmup
        return float(np.exp(-5.0 * phase * phase))


def zero_cosine_rampdown(current, epochs):
    return float(.5 * (1 + np.cos(current * np.pi / epochs)))


def test_model(model: HODDetector, epoch, valid_loader_id, wandb_run: wandb.wandb_sdk.wandb_run.Run, valid_id_best, valid_pent, 
               valid_loader_ood, model_dir):
    # eval_epoch()
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(valid_loader_id), ncols=70) as _tqdm:
            _tqdm.set_description(f'Validating: e{epoch + 1}')
            probs_i, labels_i = [], []
            valid_loss = 0
            for data, label in valid_loader_id:
                data = torch.tensor(data).to('cuda').float()
                label = torch.tensor(label).to('cuda')
                

                feature, logits, probs = model(data)
                
                # min_distance = torch.min(- distance, dim=1)                
                loss = model.criterion(logits, label)
                valid_loss += loss.sum()
                
                labels_i.append(label)
                probs_i.append(probs)
                _tqdm.update(1)

    
    valid_loss /= len(valid_loader_id)
    if wandb_run:
        wandb_run.log({'valid_loss': valid_loss})
    # summary_writer.add_scalars('Loss', {'valid_loss': valid_loss}, epoch)

    probs_i = torch.cat(probs_i, dim=0)
    labels_i = torch.cat(labels_i, dim=0)
    
    result = valid_id(probs_i, labels_i)
    print('current metric: ', end='')
    for key in result.keys():
        current = result[key]
        if valid_id_best.__contains__(key):
            if current >= valid_id_best[key]['value']:
                valid_id_best[key]['value'] = current
                valid_id_best[key]['epoch'] = epoch
                torch.save(model, f'{model_dir}/best_{key}_distance.pth')
                
                print(f'best {key} model saved in epoch {epoch}!')
            best_value = valid_id_best[key]['value']
            best_epoch = valid_id_best[key]['epoch']
            print(f'{key}: {current:.4f}({best_value:.4f} in {best_epoch})', end='')

    # eval_ood
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(valid_loader_ood), ncols=70) as _tqdm:
            _tqdm.set_description(f'Validating: ood in {epoch + 1}')
            probs_o, labels_o = [], []
            valid_loss = 0
            for data, label in valid_loader_ood:
                data = torch.tensor(data).to('cuda').float()
                label = torch.tensor(label).to('cuda')
                
                _, logits, probs = model(data)

                probs_o.append(probs)
                labels_o.append(label)
                _tqdm.update(1)
                
        probs_o = torch.cat(probs_o, dim=0)
        labels_o = torch.cat(labels_o, dim=0)

    labels_i_np = labels_i.detach().cpu().numpy()
    probs_i_np = probs_i.detach().cpu().numpy()
    print(f'valid in epoch {epoch}:')
    pre_labels_i_np = np.argmax(labels_i_np, axis=1)
    probs = np.argmax(probs_i_np, axis=1)
    print(classification_report(pre_labels_i_np, probs))

    cls_result = classification_report(pre_labels_i_np, probs, output_dict=True)
    
    if wandb_run:
        wandb_run.log({'weighted_precision': cls_result['weighted avg']['precision']})
        wandb_run.log({'weighted_recall': cls_result['weighted avg']['recall']})
        wandb_run.log({'weighted_f1_score': cls_result['weighted avg']['f1-score']})
        wandb_run.log({'macro_precision': cls_result['macro avg']['precision']})
        wandb_run.log({'macro_recall': cls_result['macro avg']['recall']})
        wandb_run.log({'macro_f1_score': cls_result['macro avg']['f1-score']})
    
    print('pent_ood_metric')
    pent_i = probs_i.detach().cpu().numpy()
    pent_o = probs_o.detach().cpu().numpy()
        
    pent_i = np.sum(np.log(pent_i) * pent_i, axis=1)
    pent_o = np.sum(np.log(pent_o) * pent_o, axis=1)
    
    result_pent = metric_ood(pent_i, pent_o)['Bas']
    print('AUROC', result_pent['AUROC'])
    
    assert result_pent['AUROC'] > 0
    
    if wandb_run:
        wandb_run.log({'AUROC': result_pent['AUROC']})


    if result_pent['AUROC'] >= valid_pent['value']:
        valid_pent['value'] = result_pent['AUROC']
        valid_pent['epoch'] = epoch
        torch.save(model, f'{model_dir}/best_ent_auroc.pth')
        print(f'best ent model saved in epoch {epoch}!')


def train_model(model: HODDetector, epoch, train_loader: DataLoader, optimizer, wandb_run: wandb.wandb_sdk.wandb_run.Run, max_epochs):
    # train_epoch()
    model.train()

    with tqdm(total=len(train_loader), ncols=80, colour='green') as _tqdm:
        _tqdm.set_description(f'Training: e{epoch + 1}')
        train_loss = 0
        for data, label in train_loader:
            data = torch.tensor(data).to('cuda').float()
            label = torch.tensor(label).to('cuda')
            
            _, logits, _ = model(data)
            loss = model.criterion(logits, label)
            # _, preds = torch.max(distance, 1)
            
            train_loss += loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            _tqdm.set_postfix(loss='{:.6f}'.format(loss.item()))
            _tqdm.update(1)
            
        train_loss /= len(train_loader)
        
        if wandb_run:
            wandb_run.log({'train_loss': train_loss})
        # summary_writer.add_scalars('Loss', {'train_loss': train_loss}, epoch)

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = 1e-4 * zero_cosine_rampdown(epoch, max_epochs)


def main():
    manual_seed = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/train.yml', help='path to config')
    parser.add_argument('--name', type=str, required=True, help='experiment name', default='hod detector')
    parser.add_argument('--note', type=str, default='runs based on hod detector', help='experiment name')
    parser.add_argument('--debug', action='store_true', help='path to config')

    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.Loader)
    
    p_train_img = config['data']['train_image_dir']
    p_train_label = config['data']['train_image_label']
    ID_labels = config['model']['ID_labels']
    OOD_labels = config['model']['OOD_labels']
    class_split = (ID_labels, OOD_labels)
    num_class = len(class_split[0])
            
    batch_size = config['train']['batch_size']
    lr = float(config['train']['lr'])
    wd = float(config['train']['wd'])
    max_epochs = config['train']['max_epochs']
    save_dir = config['train']['save_dir']
    
    logs_dir = time.strftime('%Y-%m-%d_%H_%M', time.localtime())
    logs_dir = os.path.join(save_dir, logs_dir)
    
    data_dir = config['data']['cache_dir']
    # train_npy = os.path.join(data_dir, 'data.npy')
    # label_npy = os.path.join(data_dir, 'label.npy')
    
    # make wandb
    if not args.debug:
        wandb.login()
        run = wandb.init(
            job_type='train',
            project='out of distribution',
            name=args.name,
            notes=args.note,
            tags=['train', 'ernn'],
            config=config
        )
        
        # define metric we are interested in
        run.define_metric('valid_loss', summary='min')
        run.define_metric('train_loss', summary='min')
        
        run.define_metric('weighted_precision', summary='max')
        run.define_metric('weighted_recall', summary='max')
        run.define_metric('weighted_f1_score', summary='max')
        
        run.define_metric('macro_precision', summary='max')
        run.define_metric('macro_recall', summary='max')
        run.define_metric('macro_f1_score', summary='max')
        
        run.define_metric('TNR', summary='max')
        run.define_metric('AUROC', summary='max')
        run.define_metric('DTACC', summary='max')
        run.define_metric('AUIN', summary='max')
        run.define_metric('AUOUT', summary='max')
    else:
        run = None
        
    isic = ISIC(img_dir=p_train_img, label_dir=p_train_label, cache_dir=data_dir, args=args)
    
    for fold in range(5):
        log_dir = os.path.join(logs_dir, f'fold_{fold}')
        tensor_dir = os.path.join(log_dir, 'tensor')
        model_dir = os.path.join(log_dir, 'models')
        check_dir(tensor_dir)
        check_dir(model_dir)

        # experiment initialization
        torch.manual_seed(manual_seed)
        train_set, valid_set_id, valid_set_ood = isic.split(fold, class_split=class_split, use_ood_during_training=True)
        
        print('train set num: {}'.format(len(train_set)))
        print('valid set num: {}'.format(len(valid_set_id)))
        print('valid ood set num: {}'.format(len(valid_set_ood)))
        
        train_loader = DataLoader(
            train_set, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=16
        )
        
        valid_loader_id = DataLoader(
            valid_set_id, 
            batch_size=batch_size,
            num_workers=16
        )
        valid_loader_ood = DataLoader(
            valid_set_ood, 
            batch_size=batch_size,
            num_workers=16
        )

        print('num_class: {}'.format(num_class))
        
        id_label_num = len(ID_labels)
        ood_label_num = len(OOD_labels)
                
        print('id class num: {}'.format(id_label_num))
        print('ood class num: {}'.format(ood_label_num))
        
        base_model = models.SimpleCNN(in_dim=3, out_dim=id_label_num + ood_label_num).to('cuda')
        model = HODDetector(
            base_model,
            num_inlier_classes=id_label_num,
            num_outlier_classes=ood_label_num
        ).to('cuda')
        
        if os.path.exists(config['train']['checkpoint']):
            state_dict = torch.load(config['train']['checkpoint'])
            model.load_state_dict(state_dict)
            
        # init for ddp
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
        
        # summary_writer = SummaryWriter(log_dir)
        metrics = ['pre', 'rec', 'f1s']
        valid_id_best = {}
        for metric in metrics:
            valid_id_best[metric] = {'value': 0,
                                    'epoch': 0}
        valid_pent = {'value': 0,
                    'epoch': 0}

        for epoch in range(max_epochs + 1):
            test_model(model, epoch, valid_loader_id, run, valid_id_best, 
                        valid_pent, valid_loader_ood, model_dir)
            
            train_model(model, epoch, train_loader, optimizer, run, max_epochs)

        if not args.debug:
            wandb.alert(
                title='Finish Training',
                text=f'finish training of "{args.name}"',
                level=wandb.AlertLevel.INFO
            )
        
if __name__ == '__main__':
    main()