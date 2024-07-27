import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GatedGraphConv
from scipy.stats import gaussian_kde
from scipy.stats import entropy
import numpy as np

SIZE = 512


class graph_loss(nn.Module):
    def __init__(self, num_layers, a):
        super(graph_loss, self).__init__()
        self.num_layers = num_layers
        self.a = a

    def forward(self, pair_now, person_1_now, person_2_now, scene_now, pair_pre, person_1_pre, person_2_pre, scene_pre):

        tensor_list_pair = [pair_now, pair_pre, person_1_pre, person_2_pre, scene_pre]
        tensor_list_A = [person_1_now, pair_pre, person_1_pre, person_2_pre, scene_pre]
        tensor_list_B = [person_2_now, pair_pre, person_1_pre, person_2_pre, scene_pre]
        tensor_list_scene = [scene_now, pair_pre, person_1_pre, person_2_pre, scene_pre]

        output_pair = self.gated_graph_convolution(tensor_list_pair)
        output_A = self.gated_graph_convolution(tensor_list_A)
        output_B = self.gated_graph_convolution(tensor_list_B)
        output_scene = self.gated_graph_convolution(tensor_list_scene)

        loss_pair_pre = information_theory_comparison(output_pair, torch.sum(pair_pre, dim=0))
        loss_A_pre = information_theory_comparison(output_A, torch.sum(person_1_pre, dim=0))
        loss_B_pre = information_theory_comparison(output_B, torch.sum(person_2_pre, dim=0))
        loss_scene_pre = information_theory_comparison(output_scene, torch.sum(scene_pre, dim=0))

        loss_pre = loss_pair_pre + loss_A_pre + loss_B_pre + loss_scene_pre

        loss_pair_now = information_theory_comparison(output_pair, torch.sum(pair_now, dim=0))
        loss_A_now = information_theory_comparison(output_A, torch.sum(person_1_now, dim=0))
        loss_B_now = information_theory_comparison(output_B, torch.sum(person_2_now, dim=0))
        loss_scene_now = information_theory_comparison(output_scene, torch.sum(scene_now, dim=0))

        loss_now = loss_pair_now + loss_A_now + loss_B_now + loss_scene_now

        return loss_pre * self.a + loss_now * (1 - self.a)

    def gated_graph_convolution(self, tensor_list):

        x = torch.stack(tensor_list, dim=0).cuda()
        x = torch.mean(x, dim=1).squeeze(dim=1)
        edge_index = torch.tensor([[1, 2, 3, 4], [0, 0, 0, 0]], dtype=torch.long).cuda()
        data = Data(x=x, edge_index=edge_index)
        # conv = GatedGraphConv(in_channels=x.size(-1), out_channels=x.size(-1), num_layers=self.num_layers).cuda()
        conv = GatedGraphConv(out_channels=x.size(-1), num_layers=self.num_layers).cuda()

        output = conv(data.x, data.edge_index)

        node1_output = output[0]

        return node1_output


def information_theory_comparison(tensor1, tensor2):
    
    data1 = tensor1.cpu().detach().numpy()  
    data2 = tensor2.cpu().detach().numpy()  

    smooth_data1 = data1 + np.random.normal(0, 0.01, size=data1.shape)
    smooth_data2 = data2 + np.random.normal(0, 0.01, size=data2.shape)

    kde1 = gaussian_kde(smooth_data1.T, bw_method='silverman') 
    kde2 = gaussian_kde(smooth_data2.T, bw_method='silverman')

    mutual_info = entropy(kde1.pdf(smooth_data1.T), kde2.pdf(smooth_data2.T))

    mutual_info_tensor = torch.tensor(mutual_info, device=tensor1.device)

    return mutual_info_tensor
