import torch
import torch.nn as nn
import torch.nn.functional as F


class GE2ELoss(nn.Module):
    def __init__(self, device=torch.device("cpu")):
        """
        GE2E Loss for speaker verification.
        Args:
            device: Torch device (e.g., cuda or cpu)
        """
        super(GE2ELoss, self).__init__()
        self.device = device

    def forward(self, embeddings, labels):
        """
        Forward pass for GE2E Loss.
        Args:
            embeddings: Tensor of shape (batch_size, embedding_dim) or (batch_size, time_steps, embedding_dim)
            labels: Tensor of shape (batch_size,)
        Returns:
            loss: Scalar loss value
        """
        # 如果 embeddings 是三維 (batch_size, time_steps, embedding_dim)，取平均作為代表向量
        if embeddings.dim() == 3:
            embeddings = embeddings.mean(dim=1)  # [batch_size, embedding_dim]

        # 檢查 embeddings 是否是正確的二維形狀
        if embeddings.dim() != 2:
            raise ValueError(f"Expected embeddings to have 2 dimensions, but got {embeddings.dim()} dimensions.")

        unique_labels = torch.unique(labels)  # 獲取唯一標籤
        total_loss = 0.0
        batch_size, embedding_dim = embeddings.size()

        for label in unique_labels:
            # 找到當前標籤的所有 embeddings
            label_indices = (labels == label).nonzero(as_tuple=True)[0]
            label_embeddings = embeddings[label_indices]

            # 分為 enrollment 和 test
            num_enrollment = len(label_embeddings) // 2
            enrollment_embeddings = label_embeddings[:num_enrollment]
            test_embeddings = label_embeddings[num_enrollment:]

            # 計算 centroid
            centroid = enrollment_embeddings.mean(dim=0, keepdim=True)  # [1, embedding_dim]

            # 正樣本相似度
            positive_similarities = F.cosine_similarity(test_embeddings, centroid)

            # 負樣本相似度
            negative_similarities = []
            for other_label in unique_labels:
                if other_label != label:
                    other_indices = (labels == other_label).nonzero(as_tuple=True)[0]
                    other_test_embeddings = embeddings[other_indices[len(other_indices) // 2 :]]

                    # 計算負樣本相似度
                    neg_sim = F.cosine_similarity(other_test_embeddings, centroid)
                    negative_similarities.append(neg_sim)

            if len(negative_similarities) > 0:
                negative_similarity_sum = torch.cat(negative_similarities).exp().sum()
            else:
                negative_similarity_sum = torch.tensor(1e-6, device=self.device)  # 避免 log(0)

            # GE2E Loss 計算
            positive_similarity_sum = positive_similarities.exp().sum()
            loss = torch.log(negative_similarity_sum) - torch.log(positive_similarity_sum)

            total_loss += loss

        return total_loss / len(unique_labels)


# 測試代碼
if __name__ == "__main__":
    # 模擬隨機輸入
    batch_size = 16
    time_steps = 10
    embedding_dim = 64

    # 隨機生成 embeddings (三維情況)
    embeddings = torch.randn(batch_size, time_steps, embedding_dim, requires_grad=True).to("cuda")
    labels = torch.randint(0, 4, (batch_size,), device="cuda")  # 隨機生成 4 種標籤

    loss_fn = GE2ELoss(device="cuda")
    loss = loss_fn(embeddings, labels)
    print(f"GE2E Loss: {loss.item():.4f}")

    # 反向傳播檢查
    loss.backward()
    print(f"Gradients computed successfully!")
