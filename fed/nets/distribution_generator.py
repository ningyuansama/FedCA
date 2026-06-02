import torch
import torch.nn as nn
import torch.nn.functional as F

# class DistributionGenerator(nn.Module):
#     def __init__(self, feat_dim, num_classes, hidden_dim=512):
#         super(DistributionGenerator, self).__init__()
#         self.feat_dim = feat_dim
#         self.num_classes = num_classes
#         self.hidden_dim = hidden_dim

#         input_dim = feat_dim + num_classes  # z + one-hot class
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_dim, feat_dim)
#         )

#     def forward(self, z, labels):
#         """
#         z: [B, feat_dim] 随机噪声
#         labels: [B] Tensor 或 list，类别索引
#         """
#         if not torch.is_tensor(labels):
#             labels = torch.tensor(labels, device=z.device)

#         # 如果 labels 是 batch size 不匹配，补齐或截断
#         if labels.size(0) != z.size(0):
#             if labels.size(0) < z.size(0):
#                 pad_len = z.size(0) - labels.size(0)
#                 labels = torch.cat([labels, torch.zeros(pad_len, device=z.device, dtype=labels.dtype)])
#             else:
#                 labels = labels[:z.size(0)]

#         y_onehot = F.one_hot(labels.long(), num_classes=self.num_classes).float()
#         # 再确保 z 和 y_onehot 在同一 device
#         y_onehot = y_onehot.to(z.device)

#         x = torch.cat([z, y_onehot], dim=1)  # [B, feat_dim + num_classes]
#         out = self.net(x)
#         return out

import torch
import torch.nn as nn

class DistributionGenerator(nn.Module):

    def __init__(self,feat_dim,num_classes,hidden_dim=512,embed_dim=128):

        super().__init__()

        self.label_emb=nn.Embedding(num_classes,embed_dim)

        input_dim=feat_dim+embed_dim

        self.net=nn.Sequential(

            nn.Linear(input_dim,hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim,hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Linear(hidden_dim,feat_dim)
        )


    def forward(self,z,labels):

        labels=labels.long()

        emb=self.label_emb(labels)

        x=torch.cat([z,emb],dim=1)

        return self.net(x)