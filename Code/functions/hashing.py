import logging
import time
import numpy as np
import torch
import copy
import configs
from utils.misc import Timer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
#from fairlearn.metrics import equalized_odds_difference

def prepare_db_dataset(config):
    logging.info('Creating Database Dataset')

    db_dataset = configs.dataset(config, filename='database.txt', transform_mode='test')
    
    return db_dataset

def get_hamm_dist(codes, centroids, margin=0, normalize=False):
    with torch.no_grad():
        nbit = centroids.size(1)
        dist = 0.5 * (nbit - torch.matmul(codes.sign(), centroids.sign().t()))

        if normalize:
            dist = dist / nbit

        if margin == 0:
            return dist
        else:
            codes_clone = codes.clone()
            codes_clone[codes_clone.abs() < margin] = 0
            dist_margin = 0.5 * (nbit - torch.matmul(codes_clone.sign(), centroids.sign().t()))
            if normalize:
                dist_margin = dist_margin / nbit
            return dist, dist_margin


def get_codes_and_labels(model, loader):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    vs = []
    ts = []
    for e, (d, t) in enumerate(loader):
        print(f'[{e + 1}/{len(loader)}]', end='\r')
        with torch.no_grad():
            # model forward
            d, t = d.to(device), t.to(device)
            v = model(d)
            if isinstance(v, tuple):
                v = v[0]

            vs.append(v)
            ts.append(t)

    print()
    vs = torch.cat(vs)
    ts = torch.cat(ts)
    return vs, ts

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def val_accuracy(query_label, ta, sa, ta_cls, sa_cls):
    with torch.no_grad():


        query_label = query_label.t()

        correct = query_label.eq(ta.view(1, -1).expand_as(query_label))

        batch_size = ta.size(0)
        
        group=[]
        group_num=[]
        for i in range(ta_cls):
            sa_group=[]
            sa_group_num=[]
            for j in range(sa_cls):
                eps=1e-8

                sa_group.append(((sa==j)*(ta==i)).float().sum() *(100 /batch_size))
                sa_group_num.append(torch.zeros(1).to(sa.device)+batch_size)
            group.append(sa_group)
            group_num.append(sa_group_num)
       
        return group,group_num
    
def compute_odds(val_loader, query_label,multiclass = False): 
    if multiclass == False:
        ta_cls = 2
        sa_cls = 2
    else:
        ta_cls = 5  #utkface_multiclass = ethnicity = 5
        sa_cls = 2


    groupAcc=[]
    for i in range(ta_cls):
        saGroupAcc=[]
        for j in range(sa_cls):
            saGroupAcc.append(AverageMeter())
        groupAcc.append(saGroupAcc)


    
    with torch.no_grad():
        query_label = torch.tensor(query_label).cuda()
        #start1 = time.time()
        for idx, (images, ta,sa) in enumerate(val_loader):
            images = images.float().cuda()
            ta = ta.cuda()
            sa=sa.cuda()
            
            ta = torch.topk(ta, 1)[1].squeeze(1)
            sa = torch.topk(sa, 1)[1].squeeze(1)
            # ta = ta.view(-1,1)
            # sa = sa.view(-1,1)
            
            
            group_acc,group_num = val_accuracy(query_label, ta, sa, ta_cls, sa_cls)
            # print('group_acc:',group_acc)
            # print('group_num:',group_num)

    
            for i in range(ta_cls):
                for j in range(sa_cls):
                    groupAcc[i][j].update(group_acc[i][j],group_num[i][j])
                    # print('group_acc:',groupAcc[i][j].val)
        # end1 = time.time()
        # print('time of update: ', end1-start1)


    
    odds=0
    odds_num=0
    for i in range(ta_cls):
        for j in range(sa_cls):
            for k in range(j+1,sa_cls):
                odds_num+=1
                # print('i,j,k:',i,j,k)
                # print('group[i][j].val: ',groupAcc[i][j].val)
                # print('group[i][k].val: ',groupAcc[i][k].val)
                # print('group[i][j].count: ',groupAcc[i][j].count)
                # print('group[i][k].count: ',groupAcc[i][k].count)
                # print('group[i][j].sum: ',groupAcc[i][j].sum)
                # print('group[i][k].sum: ',groupAcc[i][k].sum)
                # print('group[i][j].avg: ',groupAcc[i][j].avg)
                # print('group[i][k].avg: ',groupAcc[i][k].avg)
                odds+=torch.abs(groupAcc[i][j].avg-groupAcc[i][k].avg)
    odds=(odds/odds_num).item()


     
    return odds


def demographic_parity(y_true, y_pred, sensitive):
    # 计算不同敏感属性组成的人群中，预测结果的比例
    p1 = np.mean(y_pred[sensitive == 1])
    p2 = np.mean(y_pred[sensitive == 0])
    # 计算比例之差的绝对值
    dp = np.abs(p1 - p2)
    return dp


##第二种计算eo的方法
def equal_opportunity(y_true, y_pred, sensitive):
    # 计算不同敏感属性组成的正样本中，预测结果的召回率
    tp1 = np.sum((y_true == 1) & (y_pred == 1) & (sensitive == 1))
    tp2 = np.sum((y_true == 1) & (y_pred == 1) & (sensitive == 0))
    fn1 = np.sum((y_true == 1) & (y_pred == 0) & (sensitive == 1))
    fn2 = np.sum((y_true == 1) & (y_pred == 0) & (sensitive == 0))
    rec1 = tp1 / (tp1 + fn1)
    rec2 = tp2 / (tp2 + fn2)
    # 计算召回率之差的绝对值
    eo = np.abs(rec1 - rec2)
    return eo



from sklearn.metrics import confusion_matrix

def equal_odds1(y_true, y_pred, sensitive):
    # 计算不同敏感属性组成的正样本中，预测结果的混淆矩阵
    cm1 = confusion_matrix(y_true[sensitive == 1], y_pred[sensitive == 1])
    cm2 = confusion_matrix(y_true[sensitive == 0], y_pred[sensitive == 0])
    # 计算召回率和特异度
    tpr1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    tpr2 = cm2[1, 1] / (cm2[1, 0] + cm2[1, 1])
    fpr1 = cm1[0, 1] / (cm1[0, 0] + cm1[0, 1])
    fpr2 = cm2[0, 1] / (cm2[0, 0] + cm2[0, 1])
    # 计算召回率之差和特异度之差的绝对值
    eo = np.abs(tpr1 - tpr2) + np.abs(fpr1 - fpr2)
    return eo


def equal_odds2(y_true, y_pred, sensitive):
    # 计算不同敏感属性组成的正样本中，预测结果的ROC AUC
    auc1 = roc_auc_score(y_true[sensitive == 1], y_pred[sensitive == 1])
    auc2 = roc_auc_score(y_true[sensitive == 0], y_pred[sensitive == 0])
    # 计算ROC AUC之差的绝对值
    eo = np.abs(auc1 - auc2)
    return eo

def calculate_mAP(db_codes, db_labels,
                  test_codes, test_labels,
                  R, threshold=0.):
    # clone in case changing value of the original codes
    db_codes = db_codes.clone()
    test_codes = test_codes.clone()

    # if value within margin, set to 0
    if threshold != 0:
        db_codes[db_codes.abs() < threshold] = 0
        test_codes[test_codes.abs() < threshold] = 0

    # binarized
    db_codes = torch.sign(db_codes)  # (ndb, nbit)
    test_codes = torch.sign(test_codes)  # (nq, nbit)

    db_labels = db_labels.cpu().numpy()
    test_labels = test_labels.cpu().numpy()

    dist = []
    nbit = db_codes.size(1)

    timer = Timer()
    total_timer = Timer()

    timer.tick()
    total_timer.tick()

    with torch.no_grad():
        db_codes_ttd = configs.tensor_to_dataset(db_codes)
        db_codes_loader = configs.dataloader(db_codes_ttd, 32, False, 0, False)

        # calculate hamming distance
        for i, db_code in enumerate(db_codes_loader):
            dist.append(0.5 * (nbit - torch.matmul(test_codes, db_code.t())).cpu())
            timer.toc()
            print(f'Distance [{i + 1}/{len(db_codes_loader)}] ({timer.total:.2f}s)', end='\r')

        dist = torch.cat(dist, 1)  # .numpy()
        print()

    # fast sort
    timer.tick()
    # different sorting will have affect on mAP score! because the order with same hamming distance might be diff.
    # unsorted_ids = np.argpartition(dist, R - 1)[:, :R]

    # torch sorting is quite fast, pytorch ftw!!!
    topk_ids = torch.topk(dist, R, dim=1, largest=False)[1].cpu()
    timer.toc()
    print(f'Sorting ({timer.total:.2f}s)')

    # calculate mAP
    timer.tick()
    APx = []
    print('dist: ',dist.shape)
    for i in range(dist.shape[0]): # dist.shape：[2000, 21708]
        label = test_labels[i, :] # label: [0 1]
        label[label == 0] = -1    # label:[-1 1]
        idx = topk_ids[i, :]  #idx.shape: [5000] / idx:  tensor([1709, 2246, 1723,  ..., 5622, 7624, 5791])
        # idx = idx[np.argsort(dist[i, :][idx])]
        imatch = np.sum(np.equal(db_labels[idx[0: R], :], label), 1) > 0 #imatch.shape:  (5000,) / imatch:  [False False False ... False False False]
        rel = np.sum(imatch)   #rel.shape:() / rel: 2
        Lx = np.cumsum(imatch) #Lx,shape:  (5000,) / Lx:  [0 0 0 ... 2 2 2]
        Px = Lx.astype(float) / np.arange(1, R + 1, 1) #Px.shape:  (5000,) /Px:  [0. 0. 0. ... 0.00040016 0.00040008 0.0004  ]
        if rel != 0:
            APx.append(np.sum(Px * imatch) / rel)
        else:  # didn't retrieve anything relevant
            APx.append(0)
        timer.toc()

        #print(f'Query [{i + 1}/{dist.shape[0]}] ({timer.total:.2f}s)', end='\r')
        #print('idx')
    print('mAP caculated finished')
    total_timer.toc()
    logging.info(f'Total time usage for calculating mAP: {total_timer.total:.2f}s')

    return np.mean(np.array(APx))

def calculate_odds(config,db_codes, db_labels,
                  test_codes, test_labels,
                  R, threshold=0.):
    start = time.time()
    # clone in case changing value of the original codes
    db_codes = db_codes.clone()
    test_codes = test_codes.clone()

    # if value within margin, set to 0
    if threshold != 0:
        db_codes[db_codes.abs() < threshold] = 0
        test_codes[test_codes.abs() < threshold] = 0

    # binarized
    db_codes = torch.sign(db_codes)  # (ndb, nbit)
    test_codes = torch.sign(test_codes)  # (nq, nbit)

    db_labels = db_labels.cpu().numpy()
    test_labels = test_labels.cpu().numpy()

    dist = []
    nbit = db_codes.size(1)

    with torch.no_grad():
        db_codes_ttd = configs.tensor_to_dataset(db_codes)
        db_codes_loader = configs.dataloader(db_codes_ttd, 32, False, 0, False)

        # calculate hamming distance
        for i, db_code in enumerate(db_codes_loader):
            dist.append(0.5 * (nbit - torch.matmul(test_codes, db_code.t())).cpu())
            #print(f'Distance [{i + 1}/{len(db_codes_loader)}] ({timer.total:.2f}s)', end='\r')

        dist = torch.cat(dist, 1)  # .numpy()
        print()

    topk_ids = torch.topk(dist, R, dim=1, largest=False)[1].cpu()
    

    db_dataset = prepare_db_dataset(config)
    total_odds = 0
    print('Length of db dataset: ',topk_ids[0].shape)
    for i in range(dist.shape[0]): # dist.shape：[2000, 21708]
  
        label = test_labels[i, :] # label: [0 1]
        idx = topk_ids[i, :]  #idx.shape: [5000] / idx:  tensor([1709, 2246, 1723,  ..., 5622, 7624, 5791])
        ################################ caculate equalized odds: ##################################################
        data_index = np.array(idx)
        
        if config['dataset']=='utkface' or config['dataset']=='utkface_multicls':
            data = np.array(db_dataset.img_list)
            gender = np.array(db_dataset.gender_list)
            age = np.array(db_dataset.age_list)
            ethnicity = np.array(db_dataset.ethnicity_list)
            
            db_dataset.img_list = data[data_index]
            db_dataset.gender_list = gender[data_index]
            db_dataset.age_list = age[data_index]
            db_dataset.ethnicity_list = ethnicity[data_index]
        
        elif config['dataset']=='celeba':
            data = np.array(db_dataset.img_list)
            att = np.array(db_dataset.att)
            
            db_dataset.img_list = data[data_index]
            db_dataset.att = att[data_index]
        
    
        bs = min(100,idx.shape[0])
        db_loader = configs.dataloader(db_dataset, bs, shuffle=True, workers=10, drop_last=False)
        
        query_label = np.argmax(label)
        query_label = np.repeat(query_label,bs).reshape(-1,1)

        if config['dataset'] == 'utkface_multicls':
            odds = compute_odds(db_loader, query_label, multiclass=True)
        else:
            odds = compute_odds(db_loader, query_label, multiclass=False)
        
        total_odds+=odds
        
        if config['dataset']=='utkface' or config['dataset']=='utkface_multicls':
            
            db_dataset.img_list = data
            db_dataset.gender_list = gender
            db_dataset.age_list = age
            db_dataset.ethnicity_list = ethnicity
        #print('Length of db dataset: ',len(db_dataset))
        
        elif config['dataset']=='celeba':

            db_dataset.img_list = data
            db_dataset.att = att
            
        print('equalized odds:', round(odds,4))
        print(f'Query [{i + 1}/{dist.shape[0]}]', end='\r')


    total_odds /= dist.shape[0]
    end = time.time()
    print('odds caculated finished, equalized odds: ',round(total_odds,4))
    print('total time of caculate odds:', round(end-start,2))
    
    logging.info(f'Total equalized odds: {total_odds:.2f}s')
    
    return total_odds




def calculate_fairness(config,db_codes, db_labels,
                  test_codes, test_labels,
                  R, threshold=0.):
    start = time.time()
    # clone in case changing value of the original codes
    db_codes = torch.tensor(db_codes).clone()
    test_codes = torch.tensor(test_codes).clone()

    # if value within margin, set to 0
    if threshold != 0:
        db_codes[db_codes.abs() < threshold] = 0
        test_codes[test_codes.abs() < threshold] = 0

    # binarized
    db_codes = torch.sign(db_codes)  # (ndb, nbit)
    test_codes = torch.sign(test_codes)  # (nq, nbit)

    db_labels = torch.tensor(db_labels).cpu().numpy()
    test_labels = torch.tensor(test_labels).cpu().numpy()

    dist = []
    nbit = db_codes.size(1)

    with torch.no_grad():
        db_codes_ttd = configs.tensor_to_dataset(db_codes)
        db_codes_loader = configs.dataloader(db_codes_ttd, 32, False, 0, False)

        # calculate hamming distance
        for i, db_code in enumerate(db_codes_loader):
            dist.append(0.5 * (nbit - torch.matmul(test_codes, db_code.t())).cpu())
            #print(f'Distance [{i + 1}/{len(db_codes_loader)}] ({timer.total:.2f}s)', end='\r')

        dist = torch.cat(dist, 1)  # .numpy()
        print()

    topk_ids = torch.topk(dist, R, dim=1, largest=False)[1].cpu()
    
    total_odds1 = 0
    total_odds2 = 0
    total_odds3 = 0
    total_opportunities = 0
    total_dp = 0
    total_pp = 0
    db_dataset = prepare_db_dataset(config)
    print('Length of db dataset: ',topk_ids[0].shape)
    for i in range(dist.shape[0]): # dist.shape：[2000, 21708]
  
        label = test_labels[i, :] # label: [0 1]
        idx = topk_ids[i, :]  #idx.shape: [5000] / idx:  tensor([1709, 2246, 1723,  ..., 5622, 7624, 5791])
        ################################ caculate equalized odds: ##################################################
        data_index = np.array(idx)
        
        if config['dataset']=='utkface' or config['dataset']=='utkface_multicls' or config['dataset']=='utkface_multicls2':
            data = np.array(db_dataset.img_list)
            gender = np.array(db_dataset.gender_list)
            age = np.array(db_dataset.age_list)
            ethnicity = np.array(db_dataset.ethnicity_list)
            
        elif config['dataset']=='celeba':
            data = np.array(db_dataset.img_list)
            att = np.array(db_dataset.att)



        if config['dataset'] == 'utkface_multicls':
            ta = ethnicity
            sa = age
        if config['dataset'] == 'utkface_multicls2':
            ta = age
            sa = ethnicity
        elif config['dataset']=='utkface':
            ta = gender
            sa = ethnicity
        elif config['dataset']=='celeba':
            ta = att[:,3]
            sa = att[:,21]
        
        query_label = np.argmax(label)
        query_label = np.repeat(query_label,len(ta))
        
        y_pred = np.zeros_like(ta)
        y_pred[data_index] = 1
        y_true = np.equal(ta, query_label).astype(int)
        sensitive_features = sa
        
        # print('y_pred',y_pred.shape)
        # print('y_true',y_true.shape)
        # print('sensitive_features',sensitive_features.shape)
            
        odds1 = equal_odds1(y_true, y_pred, sensitive_features)
        odds2 = equal_odds2(y_true, y_pred, sensitive_features)
        #odds3 = equalized_odds_difference(y_true, y_pred, sensitive_features)
        
        opportunities = equal_opportunity(y_true, y_pred, sensitive_features)
        parities = demographic_parity(y_true, y_pred, sensitive_features)
        #pp = predictive_parity(y_true, y_pred, sensitive_features)
        
        total_odds1 += odds1
        total_odds2 += odds2
        #total_odds3 += odds3
        total_opportunities += opportunities
        total_dp += parities
        #total_pp += pp
        
        #print('equalized odds:', round(odds,4))
        print(f'Query [{i + 1}/{dist.shape[0]}]', end='\r')


    total_odds1 /= dist.shape[0]
    total_odds2 /= dist.shape[0]
    #total_odds3 /= dist.shape[0]
    total_opportunities /= dist.shape[0]
    total_dp /= dist.shape[0]
    #total_pp /= dist.shape[0]
    end = time.time()
    print('equalized odds1: ',round(total_odds1,4))
    print('equalized odds2: ',round(total_odds2,4))
    #print('equalized odds3: ',round(total_odds3,4))
    
    print('equalized opportunities: ',round(total_opportunities,4))
    print('demographic parities: ',round(total_dp,4))
    #print('predictive parities: ',round(total_pp,4))
    
    print('total time of caculate odds:', round(end-start,2))
    
    #logging.info(f'Total equalized odds: {total_odds:.2f}s')
    
    return total_odds1,total_odds2,total_opportunities,total_dp



def sign_dist(inputs, centroids, margin=0):
    n, b1 = inputs.size()
    nclass, b2 = centroids.size()

    assert b1 == b2, 'inputs and centroids must have same number of bit'

    # sl = relu(margin - x*y)
    out = inputs.view(n, 1, b1) * centroids.sign().view(1, nclass, b1)
    out = torch.relu(margin - out)  # (n, nclass, nbit)

    return out


def calculate_similarity_matrix(centroids):
    nclass = centroids.size(0)
    sim = torch.zeros(nclass, nclass, device=centroids.device)

    for rc in range(nclass):
        for cc in range(nclass):
            sim[rc, cc] = (centroids[rc] == centroids[cc]).float().mean()

    return sim
