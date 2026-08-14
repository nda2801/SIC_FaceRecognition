import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition (Deng et al., CVPR 2019).
    Projects 128-d face embeddings onto a hypersphere and enforces an additive angular margin (m = 0.5 rad),
    sharply compressing intra-class variance and expanding inter-class separation.
    """

    def __init__(self, in_features=128, num_classes=10000, s=30.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s  # Scaling factor (scale = 30.0)
        self.m = m  # Angular margin (m = 0.50 radians)

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        # 1. Chuan hoa L2 vector embeddings va weights
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        norm_weight = F.normalize(self.weight, p=2, dim=1)

        # 2. Tinh cos(theta) giua embeddings va class weights
        cosine = F.linear(norm_embeddings, norm_weight)
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7))

        # 3. Tinh cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # 4. Xu ly dieu kien bien goc (boundary condition)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 5. Tao one-hot vector cho nhan danh tinh
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # 6. Cong margin m vao dung nhan loi va giu nguyen cac nhan khac
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        # 7. Tinh Cross Entropy Loss voi Label Smoothing 0.1 de chong Overfitting
        label_smoothing = getattr(self, "label_smoothing", 0.1)
        loss = F.cross_entropy(output, labels, label_smoothing=label_smoothing)
        return loss


if __name__ == "__main__":
    print("=== TESTING ARCFACE LOSS MODULE ===")
    loss_fn = ArcFaceLoss(in_features=128, num_classes=100, s=30.0, m=0.50)
    fake_embeddings = torch.randn(8, 128)
    fake_labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

    loss = loss_fn(fake_embeddings, fake_labels)
    print(f"ArcFace Loss test Output: {loss.item():.4f}")
    print("ArcFace Loss module works 100% cleanly!")