import torch
from   torchmetrics import Metric
class MyAccuracy(Metric):
    def __init__( self):
        super().__init__()
        self.add_state("correct" , default = torch.zeros(3,3), dist_reduce_fx = "sum")
        self.add_state("total", default = torch.tensor(0), dist_reduce_fx = "sum")
    def update(self, precision):
        self.correct += precision 
        self.total   += 1

    def compute(self):
        return self.correct / self.total

def top_precision(pred, target, L_left, L_right = None ):
    precisions = {}
    T,T        = target.shape
    mask_mat   = torch.ones(T,T)
    if(L_right == None):L_right = T
    
    L_preds    = pred.tril(L_right - 1).triu(L_left)
    mask_mat   = mask_mat.tril(L_right - 1).triu(L_left)
    L_preds    = torch.squeeze(L_preds.view(1,-1))
    target     = torch.squeeze(target.view(1,-1))

    true_L = torch.tensor(T)
    for k in [1,2,5]:
        top_k          = torch.ceil(true_L/k)
        values, indics = torch.topk(L_preds, k = int(top_k))
        right_count    = torch.gather(target, -1 , indics)
        pred_count = torch.gather(L_preds, -1, indics)
        precision      = torch.sum(right_count) / top_k 
        precisions[f'precision_L{k}']  = precision

    return precisions

def all_range_presicion(pred, target):
    pred   = torch.squeeze(pred)
    target  = torch.squeeze(target)
    T,T     = target.shape
    acc_mat = torch.zeros(3,3)
    sep     = [6,12,24,T]
    for index in range(3):
        precisions = top_precision( pred  = pred
                                  , target = target
                                  , L_left = sep[index]
                                  , L_right= sep[index +1 ])
        acc_mat[index][0] = precisions[f'precision_L1']
        acc_mat[index][1] = precisions[f'precision_L2']
        acc_mat[index][2] = precisions[f'precision_L5']
    
    return acc_mat

def negative_probalbility(contact_maps, predict_prob, k = 6):

    B,T,T     = contact_maps.size()

    mask_mat   = torch.ones(T,T)
    mask_mat   = mask_mat.triu(k)
    num = 2 * torch.count_nonzero(mask_mat)

    logsoftmax = torch.nn.LogSoftmax()
    log_pred = logsoftmax(predict_prob)

    mask_c0_up = log_pred[0][0].triu(k)
    mask_c0_down = log_pred[0][0].tril(-k)
    mask_c0 = mask_c0_up + mask_c0_down
    
    mask_c1_up = log_pred[0][1].triu(k)
    mask_c1_down = log_pred[0][1].tril(-k)
    mask_c1 = mask_c1_up + mask_c1_down
    
    mask_c0 = mask_c0.unsqueeze(0)
    mask_c0 = mask_c0.unsqueeze(0)
    mask_c1 = mask_c1.unsqueeze(0)
    mask_c1 = mask_c1.unsqueeze(0)
    output = torch.cat((mask_c0, mask_c1), 1)
    target_up = torch.triu(contact_maps, diagonal = k)
    target_down = torch.tril(contact_maps, diagonal = -k)
    target = target_up + target_down

    nll = torch.nn.NLLLoss(reduction = 'sum')
    prob = nll(output, target) / num
    return prob

class MyLoss(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("loss_sum"
                       , default        = torch.tensor(0, dtype=torch.float)
                       , dist_reduce_fx = "sum")
        self.add_state("total"
                       , default        = torch.tensor(0, dtype=torch.float)
                       , dist_reduce_fx = "sum")

    def update(self, loss: torch.tensor):
        self.loss_sum += loss
        self.total    += 1

    def compute(self):
        return self.loss_sum.float() / self.total




