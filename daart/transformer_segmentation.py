# import os
# import torch
# from torch.utils.data import Dataset
# import numpy as np
# import pandas as pd
# from daart.transformer_loader import extract_patch_tokens

# class TransformerSegmentationDataset(Dataset):
#     """
#     Dataset that lazily computes ViT patch tokens for each video,
#     then splits each video into windows of length seq_len for training.
#     """
#     def __init__(self, hparams):
#         # Hyperparameters
#         self.eids       = hparams['expt_ids']
#         #self.n_datasets = len(self.eids)
#         self.video_dir  = hparams['video_dir']
#         self.config_path= hparams['transformer_config']
#         self.ckpt_path  = hparams['transformer_ckpt']
#         self.seq_len    = hparams['sequence_length']
#         self.device     = hparams.get('device', 'cuda')
#         self.labels_dir = hparams.get('labels_dir', None)

#         # Build index map: one entry per (eid, window_start)
#         import cv2
#         self.index_map = []  # list of (eid, start_frame, total_frames)
#         for eid in self.eids:
#             vf = os.path.join(self.video_dir, f"{eid}.mp4")
#             cap = cv2.VideoCapture(vf)
#             total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             cap.release()
#             n_win = (total + self.seq_len - 1) // self.seq_len
#             for w in range(n_win):
#                 start = w * self.seq_len
#                 self.index_map.append((eid, start, total))

#         # Cache for raw feats/labels per video
#         self._feat_cache = {}
#         self._label_cache= {}
#         self.input_size = None
#         self.label_names = hparams.get('label_names', [[]])[0]

#     def __len__(self):
#         return len(self.index_map)

#     def __getitem__(self, idx):
#         eid, start, total = self.index_map[idx]
#         # Load or compute full-video feats
#         if eid not in self._feat_cache:
#             vf = os.path.join(self.video_dir, f"{eid}.mp4")
#             feats = extract_patch_tokens(vf,
#                         self.config_path,
#                         self.ckpt_path,
#                         self.seq_len,
#                         device=self.device)
#             # feats: (T, C); cache transpose to (C, T)
#             self._feat_cache[eid] = torch.from_numpy(feats.T).float()
#             # load full labels too
#             lf = (os.path.join(self.labels_dir, f"{eid}_labels.csv")
#                   if self.labels_dir else
#                   os.path.join(hparams['data_dir'], 'label-hands', f"{eid}_labels.csv"))
#             df = pd.read_csv(lf)
#             self._label_cache[eid] = torch.from_numpy(
#                 np.argmax(df.values[:,1:], axis=1)
#             )
#             # set input_size once
#             if self.input_size is None:
#                 self.input_size = self._feat_cache[eid].shape[0]

#         feats_full  = self._feat_cache[eid]
#         labels_full = self._label_cache[eid]
#         # Extract window
#         end = min(start + self.seq_len, total)
#         feats_win  = feats_full[:, start:end]
#         labels_win = labels_full[start:end]
#         # If last window shorter, pad to seq_len
#         if end - start < self.seq_len:
#             pad_len = self.seq_len - (end - start)
#             feats_win  = F.pad(feats_win, (0, pad_len))
#             labels_win = F.pad(labels_win, (0, pad_len), value=0)

#         return {
#             'transformer': feats_win,
#             'labels_strong': labels_win,
#             'batch_idx': idx
#         }
