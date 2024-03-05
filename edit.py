'''
edit.py
'''


#FROM Python LIBRARY
import time
import math
import numpy as np
import psutil
import random
from collections import OrderedDict

#FROM PyTorch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image


#from iqa-pytorch
from IQA_pytorch import MS_SSIM, SSIM, GMSD, LPIPSvgg


#from this project
import backbone.vision as vision
import model
import backbone.utils as utils
import backbone.structure as structure
import backbone.module.module as module
import backbone.predefined as predefined
from backbone.utils import loadModels, saveModels, backproagateAndWeightUpdate        
from backbone.config import Config
from backbone.structure import Epoch
from dataLoader import DataLoader
from warmup_scheduler import GradualWarmupScheduler



################ V E R S I O N ################
# VERSION START (DO NOT EDIT THIS COMMENT, for tools/codeArchiver.py)

version = '1-PBVS'
subversion = '1-guided_craft'

# VERSION END (DO NOT EDIT THIS COMMENT, for tools/codeArchiver.py)
###############################################


#################################################
###############  EDIT THIS AREA  ################
#################################################


#################################################################################
#                                     MODEL                                     #
#################################################################################

class ModelList(structure.ModelListBase):
    def __init__(self):
        super(ModelList, self).__init__()


        ##############################################################
        # self.(모델이름)           :: model                   :: 필 수                     
        # self.(모델이름)_optimizer :: optimizer               :: 없어도됨
        # self.(모델이름)_scheduler :: Learning Rate Scheduler :: 없어도됨
        #-------------------------------------------------------------
        # self.(모델이름)_pretrained :: pretrained 파일 경로 :: ** /model/ 폴더 밑에 저장된 모델이 없을 시 OR optimizer 가 없을 시 ---> pretrained 경로에서 로드
        #
        # trainStep() 에서 사용 방법
        # modelList.(모델 인스턴스 이름)_optimizer
        ##############################################################
        self.NET = predefined.Guided_CRAFT(
                 in_chans=3,
                 aux_in_chans=1,
                 out_chans=1,
                 embed_dim=48,
                 depths=(2, 2, 2, 2,   2, 2, 2, 2,   2, 2, 2, 2,   2, 2, 2, 2),
                 num_heads=(6, 6, 6, 6,   6, 6, 6, 6,   6, 6, 6, 6,   6, 6, 6, 6),
                 split_size_0 = 4,
                 split_size_1 = 16,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 norm_layer=nn.LayerNorm,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv')
        self.NET_optimizer = torch.optim.RAdam(self.NET.parameters(), lr=1.e-4)
        self.NET_pretrained = "./CRAFT_SRx4.pth" # Not the same for Guided_CRAFT

        self.initApexAMP()
        self.initDataparallel()


#################################################################################
#                                     STEPS                                     #
#################################################################################

def trainStep(epoch, modelList, dataDict):
    lr_images = dataDict['LR_X16']
    eo_images = dataDict['EO']
    gt_images = dataDict['GT']


    #define loss function
    l1_criterion = nn.L1Loss()

    
    #train mode
    modelList.NET.train()

    #SR
    sr_images = modelList.NET(eo_images, lr_images)

    #calculate loss and backpropagation
    loss = l1_criterion(sr_images, gt_images)
    backproagateAndWeightUpdate(
        modelList, 
        loss, 
        modelNames=["NET"]
    )

    #return values
    lossDict = {
        'TRAIN_LOSS': loss
    }
    resultImagesDict = {
        "SR": sr_images
    }
    
    return lossDict, resultImagesDict
     


def validationStep(epoch, modelList, dataDict):
    lr_images = dataDict['LR_X16']
    eo_images = dataDict['EO']
    gt_images = dataDict['GT']

    #define loss function
    l1_criterion = nn.L1Loss()

    #eval mode
    modelList.NET.eval()

    with torch.no_grad():
        ###### SR
        sr_images = modelList.NET(eo_images, lr_images)

        #calculate loss
        loss = l1_criterion(sr_images, gt_images)

        #return values
        lossDict = {
            'VAL_LOSS': loss
        }
        resultImagesDict = {
            "SR": sr_images
        }
    
    return lossDict, resultImagesDict

def inferenceStep(epoch, modelList, dataDict):
    lr_images = dataDict['LR_X16']
    eo_images = dataDict['EO']

    #eval mode
    modelList.NET.eval()

    with torch.no_grad():
        ###### SR
        sr_images = modelList.NET(eo_images, lr_images)


    #return values
    resultImagesDict = {
        "SR": sr_images
    }
    
    return {}, resultImagesDict






#################################################################################
#                                     EPOCH                                     #
#################################################################################

modelList = ModelList()

trainEpoch = Epoch( 
                    dataLoader = DataLoader('train_tisr2024'),
                    modelList = modelList,
                    step = trainStep,
                    researchVersion = version,
                    researchSubVersion = subversion,
                    writer = utils.initTensorboardWriter(version, subversion),
                    scoreMetricDict = { 
                                    }, 
                    resultSaveData = [] ,
                    resultSaveFileName = 'train',
                    isNoResultArchiving = Config.param.save.remainOnlyLastSavedResult,
                    earlyStopIteration = Config.param.train.step.earlyStopStep,
                    name = 'TRAIN'
                    )


validationEpoch = Epoch( 
                    dataLoader = DataLoader('validation_tisr2024'),
                    modelList = modelList,
                    step = validationStep,
                    researchVersion = version,
                    researchSubVersion = subversion,
                    writer = utils.initTensorboardWriter(version, subversion),
                    scoreMetricDict = { 
                                    }, 
                    resultSaveData = [] ,
                    resultSaveFileName = 'validation',
                    isNoResultArchiving = Config.param.save.remainOnlyLastSavedResult,
                    earlyStopIteration = Config.param.train.step.earlyStopStep,
                    name = 'VALIDATION'
                    )


inferenceEpoch = Epoch( 
                    dataLoader = DataLoader('inference_tisr2024'),
                    modelList = modelList,
                    step = inferenceStep,
                    researchVersion = version,
                    researchSubVersion = subversion,
                    writer = utils.initTensorboardWriter(version, subversion),
                    scoreMetricDict = {}, 
                    resultSaveData = [] ,
                    resultSaveFileName = 'inference',
                    isNoResultArchiving = Config.param.save.remainOnlyLastSavedResult,
                    earlyStopIteration = Config.param.train.step.earlyStopStep,
                    name = 'INFERENCE'
                    )