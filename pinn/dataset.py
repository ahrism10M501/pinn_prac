import torch

def collocation_points(*args):
    grids = torch.meshgrid(*args, indexing='ij')
    coordinates = torch.stack(grids, dim=len(args)).reshape(-1, len(args))
    return coordinates

class MyData(torch.utils.data.Dataset):
    def __init__(self, cdata):
        self.cdata = cdata
        self.len  = len(cdata)

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.cdata[index]