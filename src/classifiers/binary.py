import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryClassifier(nn.Module):
    def __init__(self, h_feats):
        super(BinaryClassifier, self).__init__()
        self.h_feat = h_feats

        self.classifier = nn.Sequential(
            nn.Linear(self.h_feat, self.h_feat),
            nn.ReLU(),
            nn.Linear(self.h_feat, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)
