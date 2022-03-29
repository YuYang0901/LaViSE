import torch
import torch.nn as nn

bgr_mean = [109.5388, 118.6897, 124.6901]


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()
    if classname.find('BatchNorm1d') != -1:
        m.eval()


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class CSMRLoss(torch.nn.Module):
    """Cosine Similarity Margin Ranking Loss

    Shape:
        - output:

    """

    def __init__(self, margin=1):
        super(CSMRLoss, self).__init__()
        self.margin = torch.tensor(margin).cuda()

    def forward(self, output, target_onehot, embeddings, train_label_idx=None):
        cosine_similarity = torch.mm(output, embeddings) / \
                            torch.mm(torch.sqrt(torch.sum(output ** 2, dim=1, keepdim=True)),
                                     torch.sqrt(torch.sum(embeddings ** 2, dim=0, keepdim=True)))
        if train_label_idx is not None:
            cosine_similarity = cosine_similarity[:, train_label_idx]
            indices = torch.sum(target_onehot, dim=1) > 0
            cosine_similarity = cosine_similarity[indices]
            target_onehot = target_onehot[indices]
        false_terms = (1 - target_onehot) * cosine_similarity
        tmp = torch.sum(target_onehot * cosine_similarity, dim=1) / torch.sum(target_onehot, dim=1)
        loss = (1 - target_onehot) * (self.margin - tmp.unsqueeze(1) + false_terms)

        loss[torch.isnan(loss)] = 0.
        loss = torch.max(torch.tensor(0.).cuda(), loss.float())
        loss = torch.sum(loss, dim=1)
        loss = torch.mean(loss)
        return loss
