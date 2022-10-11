import torch
import torch.nn as nn
from torch.nn import functional as F

loss_mse = torch.nn.MSELoss()
loss_margin = torch.nn.TripletMarginLoss(margin=1.0)


class Memory(nn.Module):
    @staticmethod
    def get_update_query(mem, max_indices, score, query):
        m, d = mem.size()
        query_update = torch.zeros((m, d)).cuda()
        for i in range(m):
            idx = torch.nonzero(max_indices.squeeze(1) == i)
            a, _ = idx.size()
            if a != 0:
                query_update[i] = torch.sum(
                    ((score[idx, i] / torch.max(score[:, i])) * query[idx].squeeze(1)),
                    dim=0,
                )
            else:
                query_update[i] = 0
        return query_update

    @staticmethod
    def get_scores(mem, query):
        bs, h, w, d = query.size()
        m, d = mem.size()

        # Get a KxM correlation map between memory items and queries.
        # k, m corresponds to cosine similarity between query k and memory m
        score = torch.matmul(query, torch.t(mem))  # b X h X w X m
        score = score.view(bs * h * w, m)  # (b X h X w) X m

        score_query = F.softmax(score, dim=0)
        score_memory = F.softmax(score, dim=1)
        return score_query, score_memory

    def get_losses(self, query, keys):
        batch_size, h, w, dims = query.size()  # b X h X w X d
        softmax_score_query, softmax_score_memory = self.get_scores(keys, query)
        query_reshape = query.contiguous().view(batch_size * h * w, dims)
        _, topk_indices = torch.topk(softmax_score_memory, 2, dim=1)
        # 1st, 2nd closest memories
        nearest_indices = topk_indices[:, 0]
        second_nearest_indices = topk_indices[:, 1]
        pos = keys[nearest_indices]
        neg = keys[second_nearest_indices]

        compactness = loss_mse(query_reshape, pos.detach())
        separateness = loss_margin(query_reshape, pos.detach(), neg.detach())
        return separateness, compactness

    def read(self, query, updated_memory):
        batch_size, h, w, dims = query.size()  # b X h X w X d
        softmax_score_query, softmax_score_memory = self.get_scores(updated_memory, query)
        query_reshape = query.contiguous().view(batch_size * h * w, dims)
        concat_memory = torch.matmul(softmax_score_memory.detach(), updated_memory)  # (b X h X w) X d
        updated_query = torch.cat((query_reshape, concat_memory), dim=1)  # (b X h X w) X 2d
        updated_query = updated_query.view(batch_size, h, w, 2 * dims)
        updated_query = updated_query.permute(0, 3, 1, 2)
        return updated_query, softmax_score_query, softmax_score_memory

    def update(self, query, keys):
        batch_size, h, w, dims = query.size()  # b X h X w X d
        softmax_score_query, softmax_score_memory = self.get_scores(keys, query)
        query_reshape = query.contiguous().view(batch_size * h * w, dims)
        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)
        _, updating_indices = torch.topk(softmax_score_query, 1, dim=0)
        query_update = self.get_update_query(
            keys,
            gathering_indices,
            softmax_score_query,
            query_reshape,
        )
        updated_memory = F.normalize(query_update + keys, dim=1)
        return updated_memory.detach()

    def forward(self, query, keys, train=True):
        query = F.normalize(query, dim=1)
        query = query.permute(0, 2, 3, 1)  # b X h X w X d
        updated_query, softmax_score_query, softmax_score_memory = self.read(query, keys)
        separateness, compactness = self.get_losses(query, keys)
        if train:
            updated_memory = self.update(query, keys)
        else:
            # TODO: updated_memory is detached; can we just always update, and the caller decides whether to discard
            #       the updated memory and keep the previous one if frame is anomalous?
            updated_memory = keys
        return (
            updated_query,
            updated_memory,
            softmax_score_query,
            softmax_score_memory,
            separateness,
            compactness,
        )
