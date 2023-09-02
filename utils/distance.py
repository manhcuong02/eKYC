import numpy as np
import torch


def L1_Distance(x, y, reduction = 'mean'):
    
    assert reduction in ['sum', 'mean']
    
    dis = torch.abs(x - y)
    
    if reduction == 'mean':
        return torch.mean(dis)
    
    else:
        return torch.sum(dis)
    
def Euclidean_Distance(x, y):    
    dis = torch.sqrt(
        torch.sum(torch.square(x - y))
    )
    return dis
    
def Cosine_Distance(x, y):
    
    dot_product = torch.dot(x, y)
    
    norm_x = torch.norm(x.float())
    norm_y = torch.norm(y.float())
    
    dis = dot_product/(norm_x * norm_y)
    
    return dis

def findThreshold(model_name, distance_metric):
    base_threshold = {"cosine": 0.40, "euclidean": 0.55, "L1": 0.75}

    thresholds = {
        "VGG-Face1": {"cosine": 0.40, "euclidean": 0.60, "L1": 0.86},
        "VGG-Face2": {"cosine": 0.40, "euclidean": 0.60, "L1": 0.86},
    }

    threshold = thresholds.get(model_name, base_threshold).get(distance_metric, 0.6)

    return threshold

if __name__ == '__main__':
    x = torch.rand(5)
    y = torch.rand(5)
    
    print(Cosine_Distance(x, y))
    print(L1_Distance(x, y))
    print(Euclidean_Distance(x, y))
     