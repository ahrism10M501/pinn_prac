import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

class LinearBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(LinearBlock, self).__init__()
        self.layer = weight_norm(nn.Linear(in_feature, out_feature), dim=0)

    def forward(self, x):
        x = self.layer(x)
        x = torch.tanh(x)
        return x
    
class MLP(nn.Module):
    def __init__(self, layer_list):
        super(MLP, self).__init__()
        self.input_layer = weight_norm(nn.Linear(layer_list[0], layer_list[1]), dim=0)
        self.hidden_layers = self._make_layer(layer_list[1:-1])
        self.out_layer = nn.Linear(layer_list[-2], layer_list[-1])

    def _make_layer(self, layer_list):
        layers = []
        for i in range(len(layer_list) -1):
            block = LinearBlock(layer_list[i], layer_list[i+1])
            layers.append(block)

        return nn.Sequential(*layers) # 언패킹 연산자 *
    
    def forward(self, x):
        x = torch.tanh(self.input_layer(x))
        x = self.hidden_layers(x)
        x = self.out_layer(x)
        return x
    
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)

def pinn(layer_list):
    model = MLP(layer_list)
    model.apply(weight_init) # module의 기능, 모든 하위 레이어를 훑으면서 weight_init을 실행
    return model