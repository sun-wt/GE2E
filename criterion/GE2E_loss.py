import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_curve, auc

class GE2ELoss(nn.Module):
    def __init__(self, device=torch.device("cuda"), init_alpha=10.0, init_beta=-5.0, use_alpha_beta=True):
        """
        GE2E Loss for speaker verification with optional alpha and beta scaling.
        Args:
            device: Torch device (e.g., cuda or cpu)
            init_alpha: Initial value for alpha (scaling factor for cosine similarity)
            init_beta: Initial value for beta (offset for cosine similarity)
            use_alpha_beta: Whether to use alpha and beta in scaling.
        """
        super(GE2ELoss, self).__init__()
        self.device = device
        self.use_alpha_beta = use_alpha_beta

        if self.use_alpha_beta:
            self.alpha = nn.Parameter(torch.tensor(init_alpha, device=device))  # 可訓練的 alpha 參數
            self.beta = nn.Parameter(torch.tensor(init_beta, device=device))   # 可訓練的 beta 參數
        else:
            self.alpha = None
            self.beta = None

    def forward(self, embeddings, labels):
        if embeddings.dim() == 3:
            embeddings = embeddings.mean(dim=1)  # [batch_size, embedding_dim]

        if embeddings.dim() != 2:
            raise ValueError(f"Expected embeddings to have 2 dimensions, but got {embeddings.dim()} dimensions.")

        unique_labels = torch.unique(labels)
        total_loss = 0.0

        true_labels = []  # 用於存儲真實標籤
        pred_scores = []  # 用於存儲預測分數（相似度）

        positive_weight = 1.0  # 正樣本的權重
        negative_weight = 1.0  # 負樣本的權重
        # negative_weight = 0.2  # 負樣本的權重

        for label in unique_labels:
            label_indices = (labels == label).nonzero(as_tuple=True)[0]
            label_embeddings = embeddings[label_indices]

            num_enrollment = len(label_embeddings) // 2
            enrollment_embeddings = label_embeddings[:num_enrollment]
            test_embeddings = label_embeddings[num_enrollment:]

            centroid = enrollment_embeddings.mean(dim=0, keepdim=True)  # [1, embedding_dim]

            # 正樣本相似度
            positive_similarities = F.cosine_similarity(test_embeddings, centroid)

            # 如果啟用 alpha 和 beta，則應用縮放和偏移
            if self.use_alpha_beta:
                positive_similarities = self.alpha * positive_similarities + self.beta

            true_labels.extend([1] * len(positive_similarities))  # 正樣本標籤為 1
            pred_scores.extend(positive_similarities)  # 正樣本的相似度分數

            # 負樣本相似度
            negative_similarities = []
            for other_label in unique_labels:
                if other_label != label:
                    other_indices = (labels == other_label).nonzero(as_tuple=True)[0]
                    other_test_embeddings = embeddings[other_indices[len(other_indices) // 2:]]

                    neg_sim = F.cosine_similarity(other_test_embeddings, centroid)

                    # 如果啟用 alpha 和 beta，則應用縮放和偏移
                    if self.use_alpha_beta:
                        neg_sim = self.alpha * neg_sim + self.beta

                    negative_similarities.append(neg_sim)
                    true_labels.extend([0] * len(neg_sim))  # 負樣本標籤為 0
                    pred_scores.extend(neg_sim)  # 負樣本的相似度分數

            if len(negative_similarities) > 0:
                negative_similarity_sum = torch.cat(negative_similarities).exp().sum() * negative_weight
            else:
                negative_similarity_sum = torch.tensor(1e-6, device=self.device)  # 避免 log(0)

            positive_similarity_sum = positive_similarities.exp().sum() * positive_weight
            loss = torch.log(negative_similarity_sum) - torch.log(positive_similarity_sum)

            total_loss += loss

        # **將 true_labels 和 pred_scores 轉為 GPU 張量，並移至 CPU 計算 AUC**
        true_labels = torch.tensor(true_labels, device=self.device).cpu().numpy()
        pred_scores = torch.stack(pred_scores).detach().cpu().numpy()

        # 計算 AUC，這裡是 DET 曲線的 AUC
        fpr, tpr, thresholds = roc_curve(true_labels, pred_scores)
        fnr = 1 - tpr
        det_auc = auc(fpr, fnr)

        return total_loss / len(unique_labels), det_auc
