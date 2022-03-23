import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.distributed as dist
import matplotlib.pyplot as plt
from transformers import BertForTokenClassification, BertForMaskedLM, BertTokenizerFast, BertModel, BertForPreTraining

from model import ContactPredictionHead
from modules import get_gt_contact_maps
from data import Alphabet, load_dataset, getDataset
from metrics import MyLoss, negative_probalbility, MyAccuracy, all_range_presicion
from torch.utils.tensorboard import SummaryWriter

def get_tokens_attentions(alphabet, attention_model, tokenizer, batch_protein_ids, batch_seq_strs, device_id=-1):
    device = torch.device("cuda", device_id)
    seq_tokenizer = tokenizer
    labels, seqs = load_dataset(batch_protein_ids, batch_seq_strs)
    ids = seq_tokenizer(seqs, is_split_into_words=True, return_offsets_mapping=True, truncation=True, padding=True)
    seq_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    model = attention_model.eval()
    with torch.no_grad():
        outputs = model(input_ids=seq_ids,attention_mask=attention_mask)
    attentions = outputs[-1] #L,B,H,T,T
    attentions = torch.stack(attentions) #将元素为tensor的tuple组成一个更大的tensor
    attentions = attentions.permute(1, 0, 2, 3, 4) #B, L , H, T, T
    return batch_protein_ids, seq_ids, attentions

def train(run_path, cph_model=None, attention_model=None, device_id=-1, date=None, description='', loss_type='negloss', do_eval=False):
    #DDP backend初始化
    #torch.cuda.set_device(DEVICE)
    #dist.init_process_group(backend='nccl')
    device = torch.device("cuda", device_id)

    log_dir = os.path.join(run_path, date, description)
    save_dir = os.path.join(run_path, date, description)
    writer = SummaryWriter(log_dir)
    
    alphabet = Alphabet(0, 2, 3)
    layers = 30
    C = 16
    if cph_model is not None:
    #if dist.get_rank()==0 and model_path is not None:
        model.load_state_dict(torch.load(cph_model)) 
    else:
        model = ContactPredictionHead(
            layers * C, #30*16
            alphabet.prepend_bos,
            alphabet.append_eos,
            eos_idx=alphabet.eos_idx).to(device) 
    #model = DDP(model, device_ids=[DEVICE], output_device=DEVICE)

    tokenizer = BertTokenizerFast.from_pretrained(attention_model, do_lower_case=False)
    atten_model = BertForMaskedLM.from_pretrained(attention_model, output_attentions=True).to(device)

    #initialization
    num_epoch = 200
    batch_size = 1
    l1_alpha = 0.15
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
    train_loss = MyLoss().to(device)
    train_log_loss = MyLoss().to(device)
    train_l1_loss = MyLoss().to(device)
    valid_loss  = MyLoss().to(device)
    valid_log_loss =  MyLoss().to(device)
    valid_l1_loss = MyLoss().to(device)
    train_acc   = MyAccuracy().to(device)
    valid_acc   = MyAccuracy().to(device)   

    #获得data
    train_dataset, val_dataset = getDataset()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def validate():
        val_log_loss_mean = 0
        val_l1_loss_mean = 0
        val_loss_mean = 0
        val_acc_mean = torch.zeros(3,3)
        
        model.eval()

        for batch_idx, (batch_protein_ids, batch_seq_strs, batch_coords_dict) in enumerate(val_loader):
            batch_protein_ids, input_tokens, input_attentions = get_tokens_attentions(alphabet, atten_model, tokenizer, batch_protein_ids, batch_seq_strs, device_id=device_id)
            batch_coords = batch_coords_dict['coord']
            targets = get_gt_contact_maps(batch_coords).to(device)
            targets = targets.long().to(device) #损失函数要求target为long类型， input不做要求
            outputs_class1 = model(input_tokens, input_attentions)
            outputs_class1 = outputs_class1.unsqueeze(1) #output: B x C x L x L
            outputs_class0 = 1-outputs_class1
            outputs = torch.cat((outputs_class0, outputs_class1), 1)
            
            val_log_loss = negative_probalbility( predict_prob = outputs, contact_maps= targets, k = 6)
            for module in model.modules():
                if type(module) is nn.Linear:
                    val_l1_loss = (torch.abs(module.weight).sum()) / l1_alpha
            if loss_type == 'negloss':
                val_loss = val_log_loss
            elif loss_type == 'negl1loss':
                val_loss = val_log_loss +val_l1_loss
            
            val_acc  = all_range_presicion( pred = outputs_class1.cpu(), target = targets.cpu()) 
            valid_loss(val_loss)
            valid_l1_loss(val_l1_loss)
            valid_log_loss(val_log_loss) 
            valid_acc(val_acc.to(device))

        val_loss_mean = valid_loss.compute()
        val_l1_loss_mean = valid_l1_loss.compute()
        val_log_loss_mean = valid_log_loss.compute()
        val_acc_mean  = valid_acc.compute()
        valid_loss.reset()
        valid_l1_loss.reset()
        valid_log_loss.reset()
        valid_acc.reset()
        return val_loss_mean, val_log_loss_mean, val_l1_loss_mean, val_acc_mean

    for epoch in range(num_epoch):
        train_log_loss_mean = 0
        train_l1_loss_mean = 0
        train_loss_mean = 0
        train_acc_mean = torch.zeros(3,3)
        #设置sampler的epoch
        #train_loader.sampler.set_epoch(epoch)
        model.train()
        for batch_idx, (batch_protein_ids, batch_seq_strs, batch_coords_dict) in enumerate(train_loader):
            batch_protein_ids, input_tokens, input_attentions = get_tokens_attentions(alphabet, atten_model, tokenizer, batch_protein_ids, batch_seq_strs, device_id=device_id)
            batch_coords = batch_coords_dict['coord']
            targets = get_gt_contact_maps(batch_coords).to(device)
            assert targets[0].equal(targets[0].t())

            #forward
            outputs_class1 = model(input_tokens, input_attentions)
            outputs_class1 = outputs_class1.unsqueeze(1) #output: B x C x L x L
            outputs_class0 = 1-outputs_class1
            outputs = torch.cat((outputs_class0, outputs_class1), 1)
            
            targets = targets.long().to(device) #损失函数要求target为long类型， input不做要求
            log_loss = negative_probalbility( predict_prob = outputs, contact_maps= targets, k = 6)
            train_log_loss(log_loss)
            for module in model.modules():
                if type(module) is nn.Linear:
                    l1_loss = (torch.abs(module.weight).sum()) / l1_alpha
                    train_l1_loss(l1_loss)
            if loss_type == 'negloss':
                loss = log_loss  
            elif loss_type == 'negl1loss':
                loss = log_loss + l1_loss
            train_loss(loss)

            acc  = all_range_presicion( pred = outputs_class1.cpu(), target = targets.cpu()) 
            train_acc(acc.to(device))
            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Log_loss: {:.6f} L1_loss: {:.6f} Acc: {}'.format(
                        epoch+1, num_epoch, (batch_idx+1) * len(input_tokens), len(train_loader.dataset),
                            100. * (batch_idx+1) / len(train_loader), loss.item(), log_loss.item(), l1_loss.item(), acc))
        train_loss_mean = train_loss.compute()
        train_log_loss_mean = train_log_loss.compute()
        train_l1_loss_mean = train_l1_loss.compute()
        train_acc_mean  = train_acc.compute()
        train_loss.reset()
        train_log_loss.reset()
        train_l1_loss.reset()
        train_acc.reset()

        train_acc_dict = {}
        rangeL = ['middle','long','super long']
        topK   = ['L','L/2','L/5']

        for l_index in range(len(rangeL)):
            for k_index in range(len(topK)):
                train_acc_dict[f'{rangeL[l_index]}_{topK[k_index]}'] = train_acc_mean[l_index][k_index]

        if(do_eval is True):
            with torch.no_grad():
                valid_loss_mean, valid_log_loss_mean, valid_l1_loss_mean, valid_acc_mean = validate()
                valid_acc_dict = {}
                for l_index in range(len(rangeL)):
                    for k_index in range(len(topK)):
                        valid_acc_dict[f'{rangeL[l_index]}_{topK[k_index]}'] = valid_acc_mean[l_index][k_index]
        
        writer.add_scalars('Loss/train/loss', {'log_loss':train_log_loss_mean, 'l1_loss':train_l1_loss_mean, 'epoch_loss':train_loss_mean}, epoch)
        writer.add_scalars('Accuracy/train', train_acc_dict, epoch)
        if(do_eval is True):
            print(f'loss : train {train_loss_mean} valid: {valid_loss_mean}')
            print(f'acc  : train {train_acc_mean}  valid: {valid_acc_mean} ')
            writer.add_scalars('Loss/validate/loss', {'log_loss':valid_log_loss_mean, 'l1_loss':valid_l1_loss_mean, 'epoch_loss':valid_loss_mean}, epoch)
            writer.add_scalars('Accuracy/validate', valid_acc_dict, epoch)
        else:
            print(f'loss : train {train_loss_mean}')
            print(f'acc  : train {train_acc_mean} ')
    #if dist.get_rank()==0:
    torch.save(model.state_dict(), os.path.join(save_dir,'model.pth'))
    print("saved model")
    torch.save(optimizer.state_dict(), os.path.join(save_dir,'optimizer.pth'))
    print("saved optimizer")
    
if __name__ == '__main__':
    torch.set_num_threads(5) 
    pdb_path = '/data/pdb'
    fasta_path = '/data/fasta'
    #fasta_path = '/ProtTran/contact/test_data'
    mmcif_path = '/data/mmCIF/'
    #attention_model1 = "Rostlab/prot_bert_bfd"
    attention_model1 = '/download/prot_bert_bfd' #原始pbb
    attention_model2= '/ProtTran/StrucBERT/bc30out/prot_bert_bfd_seqstruc' #用bc30微调的pbb
    attention_model3= '/ProtTran/StrucBERT/bc100out/prot_bert_bfd_seqstruc' #用bc100微调的pbb

    run_path = '/ProtTran/contact'
    DEVICE = 1 #1-2,2-0,0-1
    LOSS_TYPE = 'negloss'
    train(run_path=run_path, attention_model=attention_model3, device_id=DEVICE, date = '0323del', description=+LOSS_TYPE, loss_type=LOSS_TYPE, do_eval=True)
    

    