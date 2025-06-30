"""
From https://github.com/vlukiyanov/pt-dec
"""

import torch
import torch.nn as nn
from typing import Tuple
from torch.nn import Parameter
from typing import Optional
from torch.nn.functional import softmax



class DEC(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        hidden_dimension: int,
        encoder: torch.nn.Module,
        alpha: float = 1.0,
        orthogonal=True,
        freeze_center=True, project_assignment=True
    ):
        """
        Module which holds all the moving parts of the DEC algorithm, as described in
        Xie/Girshick/Farhadi; this includes the AutoEncoder stage and the ClusterAssignment stage.

        :param cluster_number: number of clusters
        :param hidden_dimension: hidden dimension, output of the encoder
        :param encoder: encoder to use
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        """
        super(DEC, self).__init__()
        self.encoder = encoder
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = ClusterAssignment(cluster_number, self.hidden_dimension)

        self.loss_fn = nn.KLDivLoss(size_average=False)

    def forward(self, batch: torch.Tensor, cluster_centers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the cluster assignment using the ClusterAssignment after running the batch
        through the encoder part of the associated AutoEncoder module.

        :param batch: [batch size, embedding dimension] FloatTensor
        :return: [batch size, number of clusters] FloatTensor
        """
        
        # assert 2==3
        node_num = batch.size(0)
        batch_size = 1

        # [batch size, embedding dimension]
        # flattened_batch = batch.view(batch_size, -1)
        encoded = self.encoder(batch)
        # print(encoded.size())
        # encoded = batch
        # [batch size * node_num, hidden dimension]
        # encoded = encoded.view(batch_size * node_num, -1)
        # [batch size * node_num, cluster_number]
        assignment = self.assignment(encoded,cluster_centers)
        # [batch size, node_num, cluster_number]
        assignment = assignment.view(batch_size, node_num, -1)
        # [batch size, node_num, hidden dimension]
        encoded = encoded.view(batch_size, node_num, -1)
        # Multiply the encoded vectors by the cluster assignment     to get the final node representations
        # [batch size, cluster_number, hidden dimension]
        node_repr = torch.bmm(assignment.transpose(1, 2), encoded)
        return node_repr, assignment
    


class ClusterAssignment(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        embedding_dimension: int,
        alpha: float = 1.0,
        project_assignment=True
    ) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.

        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.project_assignment = project_assignment
        
    @staticmethod
    def project(u, v):
        return (torch.dot(u, v)/torch.dot(u, u))*u

    def forward(self, batch: torch.Tensor, cluster_centers: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.

        :param batch: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """       
        if self.project_assignment:
            cluster_centers = cluster_centers.squeeze(0)
            # print(batch.size(), cluster_centers.size())

            assignment = batch@cluster_centers.T #[N, emb_dim//cluster_number]@[emb_dim//cluster_number, cluster_number] = [N, emb_dim//cluster_number]
            # prove
            assignment = torch.pow(assignment, 2) #[N, cluster_number]
            # print(assignment.size())

            norm = torch.norm(cluster_centers, p=2, dim=-1) 
            soft_assign = assignment/norm 
            soft_assign = softmax(soft_assign, dim=-1) # [N, cluster_number]
            
            return soft_assign

        else:
            norm_squared = torch.sum(
                (batch.unsqueeze(1) - cluster_centers) ** 2, 2)
            numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
            power = float(self.alpha + 1) / 2
            numerator = numerator ** power
            return numerator / torch.sum(numerator, dim=1, keepdim=True)

