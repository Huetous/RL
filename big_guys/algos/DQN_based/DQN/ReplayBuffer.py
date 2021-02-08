import torch
import numpy as np
from collections import namedtuple
from heapq import heappush

Stats = namedtuple("Stats", ["EpRet", "EpLen"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class UniformReplayBuffer:
    def __init__(self, size, obs_dim, k_frames):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.uint8)
        self.act_buf = np.zeros(size, dtype=np.int)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.bool)

        self.max_size, self.size, self.ptr = size, 0, 0
        self.k_frames = k_frames

    def __len__(self):
        return self.size

    def store_obs(self, obs):
        assert obs is not None
        self.obs_buf[self.ptr] = obs
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def store_effect(self, a, r, d):
        self.act_buf[self.ptr - 1 % self.max_size] = a
        self.rew_buf[self.ptr - 1 % self.max_size] = r
        self.done_buf[self.ptr - 1 % self.max_size] = d

    def sample_batch(self, batch_size):
        assert self.size > batch_size > 0
        idxs = np.random.randint(0, self.size, size=batch_size)
        return self._sample_batch(idxs)

    def _sample_batch(self, buf_idxs, heap_idxs=None):
        batch = dict(obs=np.concatenate([self._get_context(i) for i in buf_idxs], 0),
                     act=self.act_buf[buf_idxs],
                     rew=self.rew_buf[buf_idxs],
                     obs2=np.concatenate([self._get_context(i + 1) for i in buf_idxs], 0),
                     done=self.done_buf[buf_idxs])

        # Is used to update priorities when using rank-based prioritized replay
        if heap_idxs is not None:
            batch["heap_idx"] = heap_idxs

        return {k: torch.as_tensor(v, device=device, dtype=torch.long if k is "act" else torch.float32)
                for k, v in batch.items()}

    def get_recent_context(self):
        assert self.size > 0
        return self._get_context(self.ptr - 1).to(device)

    def _get_context(self, i):
        end = i + 1 if i + 1 <= self.max_size else i  # extreme cases are considered
        start = end - self.k_frames

        # if there are not enough frames for contest in the buffer
        # and buffer is not full
        if start < 0 and self.size != self.max_size:
            start = 0

        # if frames in the range from start to end belong to different episodes
        for i in range(start, end - 1):
            if self.done_buf[i % self.max_size]:
                start = i + 1

        missing_context = self.k_frames - (end - start)  # how many frames are missing
        # if buffer is full but we still don't have enough frames
        # we create dummy frames that consist of zeros
        if start < 0 or missing_context > 0:
            frames = np.zeros((self.k_frames, *self.obs_buf[0].shape))
            for i in range(end - start):
                frames[i + missing_context] = self.obs_buf[i % self.max_size]
        else:
            frames = self.obs_buf[start:end]

        h, w = self.obs_buf[0].shape[0:2]
        return torch.tensor(frames.reshape(-1, h, w), dtype=torch.float32).unsqueeze(0) / 255.0


# # paper - https://arxiv.org/pdf/1511.05952.pdf
# # TODO: how to maintain heap index of each transition, how to override elements in heap?
# class RankBasedPrioritizedReplay(UniformReplayBuffer):
#     def __init__(self, size, obs_dim, k_frames, alpha):
#         super().__init__(size, obs_dim, k_frames)
#         self.alpha = alpha
#         self.max_priority = -1
#         self.heap = []
#
#     def store_obs(self, obs):
#             index = self.ptr
#         super()._store_obs(obs)
#         heappush(self.heap, [self.max_priority, heap_index, self.ptr])
#         super()._update_ptr()
#
#     def _sample_heap(self, batch_size):
#         segment_length = self.size // batch_size  # length of each segment
#         # shift each index so it is inside its corresponding segment
#         shifts = [i * segment_length for i in range(batch_size)]
#
#         partitions = np.random.randint(0, segment_length, batch_size)  # random partitions within each segment
#         return (partitions + shifts).astype(np.int)  # heap indexes from each segment
#
#     def sample_batch(self, batch_size):
#         assert self.size > batch_size > 0
#
#         heap_idxs = self._sample_heap(batch_size)
#
#         # list of replay buffer indexes that correspond to particular transitions
#         buf_idxs = [self.heap[i].index for i in heap_idxs]
#         return super()._sample_batch(buf_idxs, heap_idxs)
#
#     def update_priorities(self, changes):
#         for index, new_priority in changes:
#             new_priority = new_priority.item()
#             self.max_priority = max(self.max_priority, new_priority)
#             self.heap[index.int().item()][0] = new_priority  # heap[i] = [priority, index_in_buf]

