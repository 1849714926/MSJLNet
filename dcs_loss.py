import torch
from torch import nn


class DCSLoss(nn.Module):
    def __init__(self, k_size, margin1=0, margin2=0.7):
        super(DCSLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.k_size = k_size
        self.ranking_loss = nn.MarginRankingLoss(margin=margin2)

    def l2_normalize(self, x):
        norm = x.norm(p=2, dim=1, keepdim=True)
        return x.div(norm.expand_as(x))

    def forward(self, inputs, targets):
        n = inputs.size(0)

        # 对输入数据进行 L2 归一化
        inputs_normalized = self.l2_normalize(inputs)

        # 计算中心点
        centers = []
        for i in range(n):
            centers.append(inputs_normalized[targets == targets[i]].mean(0))
        centers = torch.stack(centers)

        # 对中心点也进行 L2 归一化
        centers_normalized = self.l2_normalize(centers)

        # 继续原来的损失计算...
        dist_pc = (inputs_normalized - centers_normalized) ** 2
        dist_pc = dist_pc.sum(1)
        dist_pc = dist_pc.sqrt()
        dist_pc = (dist_pc - self.margin1).clamp(min=0.0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(centers, centers.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_an, dist_ap = [], []
        for i in range(0, n, self.k_size):
            dist_an.append((self.margin2 - dist[i][mask[i] == 0]).clamp(min=0.0).mean())
        dist_an = torch.stack(dist_an)

        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        loss = dist_pc.mean() + dist_an.mean()
        return loss, dist_pc.mean(), dist_an.mean()


class HCLoss(nn.Module):
    """ Hetero-center-triplet-loss-for-VT-Re-ID """

    def __init__(self, margin=0.3):
        super().__init__()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, labels):
        """
            Args:
            - inputs: feature with shape (batch_size, feat_dim)
            - labels: ground truth labels with shape (batch_size)
        """
        label_uni = labels.unique()
        targets = torch.cat([label_uni, label_uni])
        label_num = len(label_uni)
        feat = feats.chunk(label_num * 2, 0)
        center = []
        for i in range(label_num * 2):
            center.append(torch.mean(feat[i], dim=0, keepdim=True))
        inputs = torch.cat(center)

        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        return loss
