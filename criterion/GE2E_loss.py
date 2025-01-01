import tensorflow as tf
import numpy as np

def cosine_similarity(a, b, axis=-1):
    """
    計算餘弦相似度。
    """
    a_norm = tf.nn.l2_normalize(a, axis=axis)
    b_norm = tf.nn.l2_normalize(b, axis=axis)
    return tf.reduce_sum(a_norm * b_norm, axis=axis)

class GE2ELoss(tf.keras.losses.Loss):
    def __init__(self, name="GE2ELoss"):
        super().__init__(name=name)

    def call(self, embeddings, labels):
        """
        embeddings: Tensor of shape (batch_size, embedding_dim)
        labels: Tensor of shape (batch_size, )
        """
        unique_labels, _ = tf.unique(labels)  # 獲取唯一的關鍵字
        total_loss = 0.0

        for label in unique_labels:
            # 找到當前關鍵字的所有 embeddings
            label_indices = tf.where(labels == label)[:, 0]
            label_embeddings = tf.gather(embeddings, label_indices)

            # 分為 enrollment 和 test
            num_enrollment = label_embeddings.shape[0] // 2
            enrollment_embeddings = label_embeddings[:num_enrollment]
            test_embeddings = label_embeddings[num_enrollment:]

            # 計算當前關鍵字的 centroid
            centroid = tf.reduce_mean(enrollment_embeddings, axis=0, keepdims=True)

            # 計算正樣本相似度
            positive_similarities = []

            # 對每個 test_embedding 計算與 centroid 的相似度，取指數後累加
            for test_embedding in test_embeddings:
                sim = tf.exp(cosine_similarity(tf.expand_dims(test_embedding, axis=0), centroid))
                # print(sim)
                positive_similarities.append(sim)

            # 將所有正樣本相似度相加
            positive_similarity_sum = tf.reduce_sum(tf.concat(positive_similarities, axis=0))
            # print('total sim:', positive_similarity_sum)

            # 計算負樣本相似度
            negative_similarities = []

            # 對於其他關鍵字
            for other_label in unique_labels:
                if other_label != label:
                    # 找到其他關鍵字的所有 embeddings
                    other_indices = tf.where(labels == other_label)[:, 0]
                    other_test_embeddings = tf.gather(embeddings, other_indices[num_enrollment:])

                    # 對其他關鍵字的每個 test_embedding 計算與當前關鍵字 centroid 的相似度
                    for other_test_embedding in other_test_embeddings:
                        neg_sim = tf.exp(cosine_similarity(tf.expand_dims(other_test_embedding, axis=0), centroid))
                        negative_similarities.append(neg_sim)

            # 合併所有負樣本相似度
            negative_similarity_sum = tf.reduce_sum(tf.concat(negative_similarities, axis=0))

            # GE2E Loss 計算，基於公式 (4)
            loss = tf.math.log(negative_similarity_sum) - tf.math.log(positive_similarity_sum)

            # 打印每個關鍵字的損失
            print(f"Keyword: {label.numpy()}, Loss: {loss.numpy():.4f}")
            total_loss += loss

        # 返回平均損失
        return total_loss / tf.cast(tf.size(unique_labels), tf.float32)


# 測試代碼
if __name__ == "__main__":
    batch_size = 16
    embedding_dim = 64

    # 模擬隨機輸入
    embeddings = tf.random.normal((batch_size, embedding_dim))
    labels = tf.convert_to_tensor(
        np.random.choice(["cat", "dog", "bird", "fish"], size=batch_size)
    )

    loss_fn = GE2ELoss()
    loss = loss_fn(embeddings, labels)
    print(f"Total GE2E Loss: {loss.numpy()}")
