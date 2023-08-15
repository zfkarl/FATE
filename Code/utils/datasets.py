import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets.folder import pil_loader
from torchvision import transforms
from PIL import Image


class HashingDataset(Dataset):
    def __init__(self, root,
                 transform=None,
                 target_transform=None,
                 filename='train',
                 separate_multiclass=False):
        self.loader = pil_loader
        self.separate_multiclass = separate_multiclass
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.filename = filename
        self.train_data = []
        self.train_labels = []

        filename = os.path.join(self.root, self.filename)

        with open(filename, 'r') as f:
            while True:
                lines = f.readline()
                if not lines:
                    break

                path_tmp = lines.split()[0]
                label_tmp = lines.split()[1:]
                self.is_onehot = len(label_tmp) != 1
                if not self.is_onehot:
                    label_tmp = lines.split()[1]
                if self.separate_multiclass:
                    assert self.is_onehot, 'if multiclass, please use onehot'
                    nonzero_index = np.nonzero(np.array(label_tmp, dtype=np.int))[0]
                    for c in nonzero_index:
                        self.train_data.append(path_tmp)
                        label_tmp = ['1' if i == c else '0' for i in range(len(label_tmp))]
                        self.train_labels.append(label_tmp)
                else:
                    self.train_data.append(path_tmp)
                    self.train_labels.append(label_tmp)

        self.train_data = np.array(self.train_data)
        self.train_labels = np.array(self.train_labels, dtype=np.float)

        print(f'Number of data: {self.train_data.shape[0]}')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.train_data[index], self.train_labels[index]
        target = torch.tensor(target)

        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.train_data)


def one_hot(nclass):
    def f(index):
        index = torch.tensor(int(index)).long()
        return torch.nn.functional.one_hot(index, nclass)

    return f

class UTKface(Dataset):
    def __init__(self,nclass,ta,sa,data_folder,transform,multiclass=0):
        self.data_folder=data_folder
        self.size = 224
        self.img_list = os.listdir(self.data_folder)
        self.img_list.sort()
        self.transform=transform
        self.att=[]
        self.ethnicity_list=[]
        self.age_list=[]
        self.gender_list=[]
        self.ta=ta
        self.sa=sa
        self.data = []
        self.nclass = nclass
        self.multiclass = multiclass

        for i in range(len(self.img_list)):
            self.gender_list.append(int(self.img_list[i].split('_')[1]=='0'))
            if multiclass == 0:
                self.age_list.append(int(self.img_list[i].split('_')[0])<35)
                self.ethnicity_list.append(int(self.img_list[i].split('_')[2]=='0'))
            elif multiclass == 1:
                self.age_list.append(int(self.img_list[i].split('_')[0])<35)
                self.ethnicity_list.append(int(self.img_list[i].split('_')[2])) 
            elif multiclass == 2:
                self.ethnicity_list.append(int(self.img_list[i].split('_')[2]=='0'))
                if int(self.img_list[i].split('_')[0])<20:
                    self.age_list.append(0)
                elif int(self.img_list[i].split('_')[0])<40:
                    self.age_list.append(1)
                elif int(self.img_list[i].split('_')[0])<60:
                    self.age_list.append(2)
                elif int(self.img_list[i].split('_')[0])<80:
                    self.age_list.append(3)
                else:
                    self.age_list.append(4)
            
        
        self.age_list = np.array(self.age_list).flatten()
        self.ethnicity_list = np.array(self.ethnicity_list).flatten()
        self.gender_list = np.array(self.gender_list).flatten()   
        self.img_list = np.array(self.img_list).flatten()   

    def ratio_preprocess(self,alpha=2):  # the sensitve group (ethnicity) has male data alhpha tiems as much as female data
        male =[]
        female =[]
        data_inedx = []
        for i in range(len(self.img_list)): #
            if self.ethnicity_list[i] == 1:
                if self.gender_list[i] == 0:
                    male.append(i)
                else:
                    female.append(i)
            else:
                data_inedx.append(i)
        ratio_female = female[:len(male)//alpha]
        data_inedx = data_inedx + male + ratio_female
        self.img_list = self.img_list[data_inedx]
        self.age_list = self.age_list[data_inedx]
        self.ethnicity_list = self.ethnicity_list[data_inedx]
        self.gender_list = self.gender_list[data_inedx]
        
        print('male nums in ethnicity: ',len(male))
        print('female nums in ethnicity: ',len(ratio_female))
        print('ratio data: ',len(self.img_list))       

    def __getitem__(self, index1):

        #index2=random.choice(range(len(self.img_list)))
        age=int(self.age_list[index1])
        gender=int(self.gender_list[index1])
        ethnicity=int(self.ethnicity_list[index1])
        
        age=torch.from_numpy(np.array(age))
        gender=torch.from_numpy(np.array(gender))
        ethnicity=torch.from_numpy(np.array(ethnicity))
        
        
        gender = torch.nn.functional.one_hot(gender, 2)
        if self.multiclass==0:
            age = torch.nn.functional.one_hot(age, 2)
            ethnicity = torch.nn.functional.one_hot(ethnicity, 2)
        elif self.multiclass==1:
            age = torch.nn.functional.one_hot(age, 2)
            ethnicity = torch.nn.functional.one_hot(ethnicity, self.nclass)
        elif self.multiclass==2:
            age = torch.nn.functional.one_hot(age, self.nclass)
            ethnicity = torch.nn.functional.one_hot(ethnicity, 2)
            
        ta=0
        sa=0

  
        img1=Image.open(self.data_folder+self.img_list[index1])
        #img2=Image.open(self.data_folder+self.img_list[index2])


        if self.ta=='gender':
            ta=gender
        elif self.ta=='age':
            ta=age
        elif self.ta=='ethnicity':
            ta=ethnicity
        
        if self.sa=="gender":
            sa=gender
        elif self.sa=="age":
            sa=age
        elif self.sa=="ethnicity":
            sa=ethnicity
    
        return self.transform(img1),ta,sa


    def __len__(self):
        return (self.img_list.shape[0])
    
    

class Celeba(Dataset):
    def __init__(self,ta,ta2,sa,sa2,data_folder,transform):
        self.data_folder=data_folder
 
        self.img_list=os.listdir(self.data_folder+'Img/img_align_celeba/')

    
        self.img_list.sort()
        self.transform=transform
        self.att=[]
        
        att_list=[]
        eval_list=[]
        with open(self.data_folder+'Anno/list_attr_celeba.txt','r') as f:
            reader=f.readlines()
            for line in reader:
                att_list.append(line.split())
        att_list=att_list[2:]

        with open(self.data_folder+'Eval/list_eval_partition.txt','r') as f:
            reader=f.readlines()
            for line in reader:
                eval_list.append(line.split())

        # print("eval_list",len(eval_list))
        # print("att_list",len(att_list))
        
        for i,eval_inst in enumerate(eval_list):
            #if eval_inst[1]==str(self.split):
            if att_list[i][0]==eval_inst[0]:
                self.att.append(att_list[i])
            else:
                pass

        
        self.att=np.array(self.att)
        self.att=(self.att=='1').astype(int)
        self.img_list=np.array(self.img_list)
        self.ta=ta
        self.ta2=ta2
        self.sa=sa
        self.sa2=sa2
        self.nclass =2

    def __getitem__(self, index1):
        
        ta=self.att[index1][int(self.ta)]
        sa=self.att[index1][int(self.sa)]

        if self.ta2!='None':
            ta2=self.att[index1][int(self.ta2)]
            ta=ta+2*ta2

        if self.sa2!='None':
            sa2=self.att[index1][int(self.sa2)]
            sa=sa+2*sa2

        ta = torch.nn.functional.one_hot(torch.from_numpy(np.array(ta)), self.nclass)
        sa = torch.nn.functional.one_hot(torch.from_numpy(np.array(sa)), self.nclass)
        
        #index2=random.choice(range(len(self.img_list)))
 
        img1=Image.open(self.data_folder+'Img/img_align_celeba/'+self.img_list[index1])
        #img2=Image.open(self.data_folder+'Img/img_align_celeba/'+self.img_list[index2])

    
     
        return self.transform(img1),ta,sa


    def __len__(self):
        return len(self.att)

#data_folder = '/home/zf/dataset/CelebA/'
#celeba = Celeba(ta=3,ta2 ='None', sa=21, sa2= 'None',data_folder=data_folder,transform)


#utkface = UTKface(ta = 'gender',sa = 'age', data_folder ='/home/zf/dataset/utkface/UTKFace/',transform = train_transform)

def celeba( **kwargs):
    nclass = 2
    ta = 3
    sa = 21
    transform = kwargs['transform']
    fn = kwargs['filename']
    reset = kwargs['reset']

    celeba = Celeba(ta=3,ta2 ='None', sa=21, sa2= 'None',data_folder='/home/zf/dataset/CelebA/',transform = transform)
    
    data = celeba.img_list
    att = celeba.att
    targets = att[:,ta]

    path = f'/home/zf/dataset/CelebA/{fn}'

    load_data = fn == 'train.txt'
    load_data = load_data and (reset or not os.path.exists(path))

    if not load_data:
        print(f'Loading {path}')
        data_index = torch.load(path)
    else:
        train_data_index = []
        query_data_index = []
        db_data_index = []

        data_id = np.arange(data.shape[0])  # [0, 1, ...]

        for i in range(nclass):
            class_mask = targets == i
            index_of_class = data_id[class_mask].copy()  # index of the class [2, 10, 656,...]
            np.random.shuffle(index_of_class)

            query_n = 250  # // (nclass // 10)

            index_for_query = index_of_class[:query_n].tolist()
            index_for_db = index_of_class[query_n:].tolist()
            index_for_train = index_for_db

            train_data_index.extend(index_for_train)
            query_data_index.extend(index_for_query)
            db_data_index.extend(index_for_db)

        train_data_index = np.array(train_data_index)
        query_data_index = np.array(query_data_index)
        db_data_index = np.array(db_data_index)
        
        print('train_data_index',train_data_index.shape)
        print('query_data_index',query_data_index.shape)
        print('db_data_index',db_data_index.shape)
        
        torch.save(train_data_index, f'/home/zf/dataset/CelebA/train.txt')
        torch.save(query_data_index, f'/home/zf/dataset/CelebA/test.txt')
        torch.save(db_data_index, f'/home/zf/dataset/CelebA/database.txt')

        data_index = {
            'train.txt': train_data_index,
            'test.txt': query_data_index,
            'database.txt': db_data_index
        }[fn]

    data = np.array(celeba.img_list)
    att = np.array(celeba.att)
    celeba.img_list = data[data_index]
    celeba.att = att[data_index]


    return celeba


def utk( **kwargs):
    nclass = 2
    transform = kwargs['transform']
    fn = kwargs['filename']
    reset = kwargs['reset']

    utkface = UTKface(nclass = 2, ta = 'gender',sa = 'ethnicity', data_folder ='/home/zf/dataset/utkface/UTKFace/',transform = transform, multiclass = 0)
    #utkface.ratio_preprocess(alpha=4)
    
    data = utkface.img_list
    gender = utkface.gender_list
    age = utkface.age_list
    ethnicity = utkface.ethnicity_list
    targets = gender

    path = f'/home/zf/dataset/utkface/{fn}'

    load_data = fn == 'train.txt'
    load_data = load_data and (reset or not os.path.exists(path))

    if not load_data:
        print(f'Loading {path}')
        data_index = torch.load(path)
    else:
        train_data_index = []
        query_data_index = []
        db_data_index = []

        data_id = np.arange(data.shape[0])  # [0, 1, ...]

        for i in range(nclass):
            class_mask = targets == i
            index_of_class = data_id[class_mask].copy()  # index of the class [2, 10, 656,...]
            np.random.shuffle(index_of_class)

            query_n = 250  # // (nclass // 10)

            index_for_query = index_of_class[:query_n].tolist()
            index_for_db = index_of_class[query_n:].tolist()
            index_for_train = index_for_db

            train_data_index.extend(index_for_train)
            query_data_index.extend(index_for_query)
            db_data_index.extend(index_for_db)

        train_data_index = np.array(train_data_index)
        query_data_index = np.array(query_data_index)
        db_data_index = np.array(db_data_index)
        
        print('train_data_index',train_data_index.shape)
        print('query_data_index',query_data_index.shape)
        print('db_data_index',db_data_index.shape)
        
        torch.save(train_data_index, f'/home/zf/dataset/utkface/train.txt')
        torch.save(query_data_index, f'/home/zf/dataset/utkface/test.txt')
        torch.save(db_data_index, f'/home/zf/dataset/utkface/database.txt')

        data_index = {
            'train.txt': train_data_index,
            'test.txt': query_data_index,
            'database.txt': db_data_index
        }[fn]

    data = np.array(utkface.img_list)
    gender = np.array(utkface.gender_list)
    age = np.array(utkface.age_list)
    ethnicity = np.array(utkface.ethnicity_list)
    utkface.img_list = data[data_index]
    utkface.gender_list = gender[data_index]
    utkface.age_list = age[data_index]
    utkface.ethnicity_list = ethnicity[data_index]

    return utkface

def utk_multicls( **kwargs):
    nclass = 5
    transform = kwargs['transform']
    fn = kwargs['filename']
    reset = kwargs['reset']

    utkface = UTKface(nclass = 5, ta = 'ethnicity',sa = 'age', data_folder ='/home/zf/dataset/utkface_multicls/utkface/UTKFace/',transform = transform, multiclass = 1)
    data = utkface.img_list
    gender = utkface.gender_list
    age = utkface.age_list
    ethnicity = utkface.ethnicity_list
    targets = ethnicity

    path = f'/home/zf/dataset/utkface_multicls/utkface/{fn}'

    load_data = fn == 'train.txt'
    load_data = load_data and (reset or not os.path.exists(path))

    if not load_data:
        print(f'Loading {path}')
        data_index = torch.load(path)
    else:
        train_data_index = []
        query_data_index = []
        db_data_index = []

        data_id = np.arange(data.shape[0])  # [0, 1, ...]

        for i in range(nclass):
            class_mask = targets == i
            index_of_class = data_id[class_mask].copy()  # index of the class [2, 10, 656,...]
            np.random.shuffle(index_of_class)

            query_n = 20  # query_n * nclass = query: 20 * 5 = 100

            index_for_query = index_of_class[:query_n].tolist()
            index_for_db = index_of_class[query_n:].tolist()
            index_for_train = index_for_db

            train_data_index.extend(index_for_train)
            query_data_index.extend(index_for_query)
            db_data_index.extend(index_for_db)

        train_data_index = np.array(train_data_index)
        query_data_index = np.array(query_data_index)
        db_data_index = np.array(db_data_index)
        
        print('train_data_index',train_data_index.shape)
        print('query_data_index',query_data_index.shape)
        print('db_data_index',db_data_index.shape)
        
        torch.save(train_data_index, f'/home/zf/dataset/utkface_multicls/utkface/train.txt')
        torch.save(query_data_index, f'/home/zf/dataset/utkface_multicls/utkface/test.txt')
        torch.save(db_data_index, f'/home/zf/dataset/utkface_multicls/utkface/database.txt')

        data_index = {
            'train.txt': train_data_index,
            'test.txt': query_data_index,
            'database.txt': db_data_index
        }[fn]

    data = np.array(utkface.img_list)
    gender = np.array(utkface.gender_list)
    age = np.array(utkface.age_list)
    ethnicity = np.array(utkface.ethnicity_list)
    utkface.img_list = data[data_index]
    utkface.gender_list = gender[data_index]
    utkface.age_list = age[data_index]
    utkface.ethnicity_list = ethnicity[data_index]

    return utkface


def utk_multicls2( **kwargs):
    nclass = 5
    transform = kwargs['transform']
    fn = kwargs['filename']
    reset = kwargs['reset']

    utkface = UTKface(nclass = 5, ta = 'age',sa = 'ethnicity', data_folder ='/home/zf/dataset/utkface_multicls2/utkface/UTKFace/',transform = transform, multiclass = 2)
    data = utkface.img_list
    gender = utkface.gender_list
    age = utkface.age_list
    ethnicity = utkface.ethnicity_list
    targets = age

    path = f'/home/zf/dataset/utkface_multicls2/utkface/{fn}'

    load_data = fn == 'train.txt'
    load_data = load_data and (reset or not os.path.exists(path))

    if not load_data:
        print(f'Loading {path}')
        data_index = torch.load(path)
    else:
        train_data_index = []
        query_data_index = []
        db_data_index = []

        data_id = np.arange(data.shape[0])  # [0, 1, ...]

        for i in range(nclass):
            class_mask = targets == i
            index_of_class = data_id[class_mask].copy()  # index of the class [2, 10, 656,...]
            np.random.shuffle(index_of_class)

            query_n = 20  # query_n * nclass = query: 20 * 5 = 100

            index_for_query = index_of_class[:query_n].tolist()
            index_for_db = index_of_class[query_n:].tolist()
            index_for_train = index_for_db

            train_data_index.extend(index_for_train)
            query_data_index.extend(index_for_query)
            db_data_index.extend(index_for_db)

        train_data_index = np.array(train_data_index)
        query_data_index = np.array(query_data_index)
        db_data_index = np.array(db_data_index)
        
        print('train_data_index',train_data_index.shape)
        print('query_data_index',query_data_index.shape)
        print('db_data_index',db_data_index.shape)
        
        torch.save(train_data_index, f'/home/zf/dataset/utkface_multicls2/utkface/train.txt')
        torch.save(query_data_index, f'/home/zf/dataset/utkface_multicls2/utkface/test.txt')
        torch.save(db_data_index, f'/home/zf/dataset/utkface_multicls2/utkface/database.txt')

        data_index = {
            'train.txt': train_data_index,
            'test.txt': query_data_index,
            'database.txt': db_data_index
        }[fn]

    data = np.array(utkface.img_list)
    gender = np.array(utkface.gender_list)
    age = np.array(utkface.age_list)
    ethnicity = np.array(utkface.ethnicity_list)
    utkface.img_list = data[data_index]
    utkface.gender_list = gender[data_index]
    utkface.age_list = age[data_index]
    utkface.ethnicity_list = ethnicity[data_index]

    return utkface