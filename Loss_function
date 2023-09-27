import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GatedGraphConv

SIZE = 512


class graph_loss(nn.Module):
    def __init__(self, num_layers):
        super(graph_loss, self).__init__()
        self.num_layers = num_layers

    def forward(self, pair_now, person_1_now, person_2_now, scene_now, pair_pre, person_1_pre, person_2_pre, scene_pre):
       
        tensor_list_pair = [pair_now, pair_pre, person_1_pre, person_2_pre, scene_pre]
        tensor_list_A = [person_1_now, pair_pre, person_1_pre, person_2_pre, scene_pre]
        tensor_list_B = [person_2_now, pair_pre, person_1_pre, person_2_pre, scene_pre]
        tensor_list_scene = [scene_now, pair_pre, person_1_pre, person_2_pre, scene_pre]

        
        output_pair = self.gated_graph_convolution(tensor_list_pair)
        output_A = self.gated_graph_convolution(tensor_list_A)
        output_B = self.gated_graph_convolution(tensor_list_B)
        output_scene = self.gated_graph_convolution(tensor_list_scene)

        loss_pair = nn.MSELoss()(output_pair, torch.mean(pair_pre, dim=0).squeeze(dim=0))
        loss_A = nn.MSELoss()(output_A, torch.mean(person_1_pre, dim=0).squeeze(dim=0))
        loss_B = nn.MSELoss()(output_B, torch.mean(person_2_pre, dim=0).squeeze(dim=0))
        loss_scene = nn.MSELoss()(output_scene, torch.mean(scene_pre, dim=0).squeeze(dim=0))

        loss = loss_pair + loss_A + loss_B + loss_scene

        return loss * 10

    def gated_graph_convolution(self, tensor_list):
       
        x = torch.stack(tensor_list, dim=0).cuda()  
        x = torch.mean(x, dim=1).squeeze(dim=1)
        edge_index = torch.tensor([[1, 2, 3, 4], [0, 0, 0, 0]], dtype=torch.long).cuda()
        data = Data(x=x, edge_index=edge_index)
        conv = GatedGraphConv(in_channels=x.size(-1), out_channels=x.size(-1), num_layers=self.num_layers).cuda()

        output = conv(data.x, data.edge_index)

        node1_output = output[0]

        return node1_output
