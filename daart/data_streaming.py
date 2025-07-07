#!/usr/bin/env python3
# daart/data_streaming_controlled.py

import os
import numpy as np
import torch
from collections import OrderedDict

from daart.data import SingleDataset, load_label_csv, compute_sequences
from daart.transformer_loader_chunks import extract_patch_tokens_chunk_gpu

# ─── NVIDIA DALI imports ───────────────────────────────────────────────────
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
# ──────────────────────────────────────────────────────────────────────────

class StreamingSingleDataset(SingleDataset):
    """
    A SingleDataset that streams video → ViT patch embeddings on-the-fly
    using NVIDIA DALI with explicit control over video and frames per index,
    and ensures corresponding labels are loaded.
    """

    def load_data(self, sequence_length: int, input_type: str, device_id=0):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        np.random.seed(42)

        torch.cuda.set_device(device_id)
        torch.empty(1).cuda(device_id)

        self.sequence_length = sequence_length
        self.input_type = input_type
        self.config_path = self.transformer_config
        self.checkpoint_path = self.transformer_ckpt
        self.max_imgs_per_pass = getattr(self, 'max_imgs_per_pass', 256)

        # 1) Load & window labels
        lbl_path = self.paths.get('labels_strong')
        if not lbl_path or not os.path.exists(lbl_path):
            raise FileNotFoundError(f"labels_strong file not found: {lbl_path!r}")
        labels_1hot, label_names = load_label_csv(lbl_path)
        labels_idx = labels_1hot.argmax(axis=1)
        lbl_seqs = compute_sequences(labels_idx, sequence_length, self.sequence_pad)
        nseq = len(lbl_seqs)

        self.data = OrderedDict()
        self.data['labels_strong'] = lbl_seqs
        self.label_names = label_names
        self.data['markers'] = [None] * nseq

        # 2) Record video path
        vid_path = self.paths.get('markers')
        if not vid_path or not os.path.exists(vid_path):
            raise FileNotFoundError(f"markers/video file not found: {vid_path!r}")
        self.video_path = vid_path

        self.input_size = None
        self.feature_names = []

        # 3) Create video and frame requests with corresponding label indices
        self.video_frame_requests = []
        for seq_idx in range(nseq):
            start_frame = seq_idx * sequence_length
            end_frame = start_frame + sequence_length
            self.video_frame_requests.append({
                "video_path": self.video_path,
                "frame_indices": list(range(start_frame, end_frame)),
                "label_sequence": self.data['labels_strong'][seq_idx]
            })

        self.nseq = len(self.video_frame_requests)

        # 4) Define the DALI GPU-only pipeline
        import tempfile

        class ControlledVideoPipeline(Pipeline):
            def __init__(self, batch_size, num_threads, device_id, video_frame_requests):
                super().__init__(batch_size, num_threads, device_id, seed=42, prefetch_queue_depth=2)
                self.video_frame_requests = video_frame_requests
                self.current_index = 0
                self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
                self._create_file_list()
        
            def _create_file_list(self):
                self.temp_file.seek(0)
                self.temp_file.truncate(0)
                for req in self.video_frame_requests:
                    start = req["frame_indices"][0]
                    num_frames = len(req["frame_indices"])
                    self.temp_file.write(f"{req['video_path']} {start} {num_frames}\n")
                self.temp_file.flush()
        
            def define_graph(self):
                video_files = fn.external_source(self.get_file_list, num_outputs=1)
                raw_frames = fn.readers.video(
                    device="gpu",
                    file_list=video_files,
                    file_list_frame_num=True,
                    sequence_length=self.get_sequence_length_per_sample(), # We need to provide a sequence length
                    dtype=types.UINT8,
                    # ... other VideoReader arguments if needed
                )
                label_sequence = fn.external_source(self.get_labels, num_outputs=1)
                return raw_frames, label_sequence
        
            def get_file_list(self):
                idx = self.current_index
                # DALI's VideoReader with file_list expects a single file list for the entire dataset
                return [self.temp_file.name]
        
            def get_sequence_length_per_sample(self):
                idx = self.current_index
                return len(self.video_frame_requests[idx]["frame_indices"])
        
            def get_labels(self):
                idx = self.current_index
                request = self.video_frame_requests[idx]
                self.current_index += 1
                return np.array(request["label_sequence"], dtype=np.int64)
        
            def reset(self):
                self.current_index = 0
                self._create_file_list()
        
            def __del__(self):
                os.remove(self.temp_file.name)
        
        # In your StreamingSingleDatasetControlled's load_data method:
        self.pipeline = ControlledVideoPipeline(
            batch_size=1,
            num_threads=1,
            device_id=device_id,
            video_frame_requests=self.video_frame_requests
        )
        self.pipeline.build()
        self.pipeline.current_index = 0 # Initialize the index for the pipeline
        
        self.dali_iter = DALIGenericIterator(
            self.pipeline,
            output_map=["frames", "labels"], # Removed "index" from output_map
            size=self.nseq,
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.DROP,
        )

        # 5) Instantiate & build pipeline
        self.pipeline = ControlledVideoPipeline(
            batch_size=1,
            num_threads=1,
            device_id=device_id,
            video_frame_requests=self.video_frame_requests
        )
        self.pipeline.build()

        self.dali_iter = DALIGenericIterator(
            self.pipeline,
            output_map=["frames", "labels", "index"],
            size=self.nseq,
            auto_reset=True,
            last_batch_policy=LastBatchPolicy.DROP,
        )

    def __len__(self):
        return self.nseq

    def __getitem__(self, idx: int) -> dict:
        try:
            batch = next(self.dali_iter)
            frames = batch[0]["frames"][0]
            labels = batch[0]["labels"][0]
            current_index = batch[0]["index"].item()

            # Verify that the index from DALI matches the requested index
            assert current_index == idx, f"DALI index ({current_index}) does not match requested index ({idx})"

            patches = extract_patch_tokens_chunk_gpu(
                frames,
                config_path=self.config_path,
                checkpoint_path=self.checkpoint_path,
                device='cuda',
                max_imgs_per_pass=600 # Adjust as needed
            )

            if self.input_size is None:
                _, PD = patches.shape
                self.input_size = PD
                self.feature_names = [f"patch_{i}" for i in range(PD)]

            return {
                'markers': patches,
                'labels_strong': torch.from_numpy(labels).long().cuda(),
                'batch_idx': current_index
            }
        except StopIteration:
            # This should ideally not happen if auto_reset=True and __len__ is correct
            self.dali_iter.reset()
            return self[idx] # Try again after reset
        except Exception as e:
            print(f"Error in __getitem__ with index {idx}: {e}")
            raise


# #!/usr/bin/env python3
# # daart/data_streaming.py

# import os
# import numpy as np
# import torch
# from collections import OrderedDict

# from daart.data import SingleDataset, load_label_csv, compute_sequences
# from daart.transformer_loader_chunks import extract_patch_tokens_chunk_gpu

# # ─── NVIDIA DALI imports ───────────────────────────────────────────────────
# import nvidia.dali.fn as fn
# import nvidia.dali.types as types
# from nvidia.dali.pipeline import Pipeline
# from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
# # ──────────────────────────────────────────────────────────────────────────
# def load_precomputed_features(idx, sequence_length, video_path):
#     """
#     Load precomputed features for a specific batch index
    
#     Parameters:
#     -----------
#     idx : int
#         Batch index
#     sequence_length : int
#         Number of frames per sequence
#     video_path : str
#         Path to the video file
    
#     Returns:
#     --------
#     np.ndarray
#         Precomputed features for the frames in this batch
#     """
#     # Extract EID from video path
#     filename = os.path.basename(video_path)
#     if 'cam0_' in filename:
#         eid = filename.split('cam0_')[-1].split('.')[0]
#     else:
#         eid = os.path.splitext(filename)[0]
    
#     # Calculate frame indices
#     start_frame = idx * sequence_length
#     end_frame = start_frame + sequence_length
    
#     # Path to precomputed features
#     features_path = f"/home/bsb2144/daart_utils/data/ibl/features-vit_cm/{eid}_labeled.npy"
    
#     # Load the data
#     if not os.path.exists(features_path):
#         raise FileNotFoundError(f"Cannot find precomputed features at {features_path}")
    
#     # Load all features and extract the relevant frames
#     all_features = np.load(features_path)
    
#     # Extract only the rows corresponding to the current batch frames
#     batch_features = all_features[start_frame:end_frame]
    
#     print(f"Loaded features for batch {idx}: EID={eid}, frames {start_frame}-{end_frame-1}, shape={batch_features.shape}")
    
#     return batch_features

# class StreamingSingleDataset(SingleDataset):
#     """
#     A SingleDataset that streams video → ViT patch embeddings on-the-fly
#     using NVIDIA DALI for GPU-accelerated frame loading and preprocessing.
#     """

#     def load_data(self, sequence_length: int, input_type: str, device_id=0):

#         torch.manual_seed(42)
#         torch.cuda.manual_seed_all(42)
#         torch.backends.cudnn.deterministic = True
#         np.random.seed(42)  # For numpy operations
            
#         # Explicit CUDA initialization (critical fix)
#         torch.cuda.set_device(device_id)
#         torch.empty(1).cuda(device_id)

#         # 1) Basic bookkeeping
#         self.sequence_length = sequence_length
#         self.input_type = input_type  # 'features' → 'markers'
#         self.config_path = self.transformer_config
#         self.checkpoint_path = self.transformer_ckpt
#         self.max_imgs_per_pass = getattr(self, 'max_imgs_per_pass', 256)

#         # 2) Load & window labels
#         lbl_path = self.paths.get('labels_strong')
#         if not lbl_path or not os.path.exists(lbl_path):
#             raise FileNotFoundError(f"labels_strong file not found: {lbl_path!r}")
#         labels_1hot, label_names = load_label_csv(lbl_path)
#         labels_idx = labels_1hot.argmax(axis=1)
#         lbl_seqs = compute_sequences(labels_idx, sequence_length, self.sequence_pad)

#         # 3) Stash labels in self.data
#         self.data = OrderedDict()
#         self.data['labels_strong'] = lbl_seqs
#         self.label_names = label_names
#         nseq = len(lbl_seqs)
#         self.data['markers'] = [None] * nseq

#         # 4) Record video path
#         vid_path = self.paths.get('markers')
#         if not vid_path or not os.path.exists(vid_path):
#             raise FileNotFoundError(f"markers/video file not found: {vid_path!r}")
#         self.video_path = vid_path

#         # 5) Defer setting input_size & feature_names
#         self.input_size = None
#         self.feature_names = []

#         # 6) Build a tiny file-list for DALI (one line per window)
#         list_file = os.path.join(os.getcwd(), f"dali_list_{id(self)}.txt")
#         with open(list_file, 'w') as f:
#             for seq_idx in range(nseq):
#                 start = seq_idx * sequence_length
#                 f.write(f"{self.video_path} {start} {sequence_length}\n")

#         # 7) Define the DALI GPU-only pipeline
#         class VideoPipeline(Pipeline):
#             def __init__(self, list_file, seq_len, batch_size=1, num_threads=16, device_id=0, seed=42):
#                 super().__init__(batch_size, num_threads, device_id, seed=42, prefetch_queue_depth=2)
#                 vr = fn.readers.video(
#                     device="gpu",
#                     file_list=list_file,
#                     file_list_frame_num=True,
#                     sequence_length=seq_len,
#                     random_shuffle=False,
#                     dtype=types.UINT8,
#                     file_list_include_preceding_frame=True
#                 )
#                 raw_frames = vr[0]
#                 resized = fn.resize(
#                     raw_frames,
#                     size=(224, 224),
#                     interp_type=types.INTERP_LINEAR,
#                     antialias=False
#                 )
#                 self.converted = fn.crop_mirror_normalize(
#                     resized,
#                     mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
#                     std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
#                     output_layout="FCHW",
#                     #dtype=types.FLOAT16
#                 )

#             def define_graph(self):
#                 return self.converted

#         # 8) Instantiate & build pipeline (GPU device explicitly initialized)
#         self.pipeline = VideoPipeline(
#             list_file,
#             self.sequence_length,
#             batch_size=1,
#             num_threads=1,
#             device_id=device_id,
#             seed=42,
#             #shuffle=False
#         )
#         self.pipeline.build()

#         self.dali_iter = DALIGenericIterator(
#             self.pipeline,
#             ['frames'],
#             size=nseq,
#             auto_reset=True,
#             last_batch_policy=LastBatchPolicy.DROP,
#             #random_shuffle=False
#         )

#     def __len__(self):
#         return len(self.data['markers'])

#     def __getitem__(self, idx: int) -> dict:
#         labels_seq = self.data['labels_strong'][idx]
#         if (not self.inference) and (labels_seq.sum() == 0):
#             return False

#         batch = next(self.dali_iter)
#         frames = batch[0]['frames'][0]

#         saved_patches = load_precomputed_features(idx, self.sequence_length, self.video_path)

#         if idx in range(10):
#             # Add this at the start of extract_patch_tokens_chunk_gpu
#             print(f"Input shape: {frames.shape}, dtype: {frames.dtype}")
#             print(f"Sample values: {frames[0,0,0,0:5]}")  # First few values

#         patches = extract_patch_tokens_chunk_gpu(
#             frames,
#             config_path=self.config_path,
#             checkpoint_path=self.checkpoint_path,
#             device='cuda',
#             max_imgs_per_pass= 600#16
#         )

#         print('patches shape', patches.shape)
#         print(f"Patches min: {patches.min().item()}, max: {patches.max().item()}, mean: {patches.mean().item()}")
#         print(patches)
#         print('')
#         print('old shape', saved_patches.shape)
#         print(f"saved_patches min: {saved_patches.min().item()}, max: {saved_patches.max().item()}, mean: {saved_patches.mean().item()}")
#         print(saved_patches)
#         print('')
#         print('')
        
#         # Print statistics to compare distributions
#         if idx in range(10):
#             print('patches shape', patches.shape)
#             print(f"Patches min: {patches.min().item()}, max: {patches.max().item()}, mean: {patches.mean().item()}")
#             print('')
#         # Add in extract_patch_tokens_chunk_gpu before returning
#         if torch.isnan(patches).any() or torch.isinf(patches).any():
#             print("WARNING: NaN or Inf values detected in patches")
        

#         #patches = patches.float()

#         if self.input_size is None:
#             _, PD = patches.shape
#             self.input_size = PD
#             self.feature_names = [f"patch_{i}" for i in range(PD)]

#         return {
#             'markers': patches,
#             'labels_strong': torch.from_numpy(labels_seq).long().cuda(),
#             'batch_idx': idx
#         }





# # #!/usr/bin/env python3
# # # daart/data_streaming.py

# # import os
# # import numpy as np
# # import torch
# # from collections import OrderedDict

# # from daart.data import SingleDataset, load_label_csv, compute_sequences
# # from daart.transformer_loader_chunks import extract_patch_tokens_chunk_gpu

# # # ─── NVIDIA DALI imports ───────────────────────────────────────────────────
# # import nvidia.dali.fn as fn
# # import nvidia.dali.types as types
# # from nvidia.dali.pipeline import Pipeline
# # from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
# # # ──────────────────────────────────────────────────────────────────────────


# # class StreamingSingleDataset(SingleDataset):
# #     """
# #     A SingleDataset that streams video → ViT patch embeddings on-the-fly
# #     using NVIDIA DALI for GPU-accelerated frame loading and preprocessing.
# #     """

# #     def load_data(self, sequence_length: int, input_type: str):
# #         # 1) Basic bookkeeping
# #         self.sequence_length   = sequence_length
# #         self.input_type        = input_type   # 'features' → 'markers'
# #         self.config_path       = self.transformer_config
# #         self.checkpoint_path   = self.transformer_ckpt
# #         self.max_imgs_per_pass = getattr(self, 'max_imgs_per_pass', 256)

# #         # 2) Load & window labels
# #         lbl_path = self.paths.get('labels_strong')
# #         if not lbl_path or not os.path.exists(lbl_path):
# #             raise FileNotFoundError(f"labels_strong file not found: {lbl_path!r}")
# #         labels_1hot, label_names = load_label_csv(lbl_path)
# #         labels_idx = labels_1hot.argmax(axis=1)
# #         lbl_seqs   = compute_sequences(labels_idx, sequence_length, self.sequence_pad)

# #         # 3) Stash labels in self.data
# #         self.data               = OrderedDict()
# #         self.data['labels_strong'] = lbl_seqs
# #         self.label_names        = label_names
# #         nseq = len(lbl_seqs)
# #         self.data['markers']    = [None] * nseq

# #         # 4) Record video path
# #         vid_path = self.paths.get('markers')
# #         if not vid_path or not os.path.exists(vid_path):
# #             raise FileNotFoundError(f"markers/video file not found: {vid_path!r}")
# #         self.video_path = vid_path

# #         # 5) Defer setting input_size & feature_names
# #         self.input_size    = None
# #         self.feature_names = []

# #         # 6) Build a tiny file-list for DALI (one line per window)
# #         list_file = os.path.join(os.getcwd(), f"dali_list_{id(self)}.txt")
# #         with open(list_file, 'w') as f:
# #             for seq_idx in range(nseq):
# #                 start = seq_idx * sequence_length
# #                 f.write(f"{self.video_path} {start} {sequence_length}\n")

# #         # 7) Define the DALI GPU-only pipeline
# #         class VideoPipeline(Pipeline):
# #             def __init__(self, list_file, seq_len, batch_size=1, num_threads=16, device_id=0):
# #                 super().__init__(batch_size, num_threads, device_id, seed=42, prefetch_queue_depth=2)
# #                 vr = fn.readers.video(
# #                     device="gpu",
# #                     file_list=list_file,
# #                     file_list_frame_num=True,
# #                     sequence_length=seq_len,
# #                     random_shuffle=False,
# #                     dtype=types.UINT8,
# #                     file_list_include_preceding_frame=True
# #                 )
# #                 raw_frames = vr[0]
# #                 resized = fn.resize(
# #                     raw_frames, 
# #                     size=(224, 224),
# #                     interp_type=types.INTERP_LINEAR,
# #                     antialias=False  # Faster resize
# #                 )

# #                 # Use half precision
# #                 self.converted = fn.crop_mirror_normalize(
# #                     resized,
# #                     mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
# #                     std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
# #                     output_layout="FCHW",
# #                     dtype=types.FLOAT16  # Use half precision
# #                 )

# #             def define_graph(self):
# #                 # returns: [batch, seq_len, C, H, W]
# #                 return self.converted

# #         # 8) Instantiate & build pipeline (always GPU, device_id=0)
# #         device_id = 0
# #         self.pipeline = VideoPipeline(
# #             list_file,
# #             self.sequence_length,
# #             batch_size=1,
# #             num_threads=1,
# #             device_id=device_id
# #         )
# #         self.pipeline.build()
# #         self.dali_iter = DALIGenericIterator(
# #             self.pipeline,
# #             ['frames'],
# #             size=nseq,
# #             auto_reset=True,
# #             last_batch_policy=LastBatchPolicy.DROP
# #         )

# #     def __len__(self):
# #         return len(self.data['markers'])

    
# #     def __getitem__(self, idx: int) -> dict:
# #         # 1) Fetch labels
# #         labels_seq = self.data['labels_strong'][idx]
# #         if (not self.inference) and (labels_seq.sum() == 0):
# #             return False

# #         # 2) Get the next batch from DALI
# #         batch = next(self.dali_iter)

# #         # 3) Get frames - already on GPU in half precision
# #         frames = batch[0]['frames'][0]  # Using index 0 since batch size is 1

# #         # 4) Process with ViT-MAE
# #         patches = extract_patch_tokens_chunk_gpu(
# #             frames,
# #             config_path=self.config_path,
# #             checkpoint_path=self.checkpoint_path,
# #             device='cuda',
# #             max_imgs_per_pass= 16  #self.max_imgs_per_pass
# #         )

# #         # 5) Convert to float32 to match the rest of the model pipeline
# #         patches = patches.float()  # Convert from float16 to float32

# #         # 6) Update metadata if needed
# #         if self.input_size is None:
# #             _, PD = patches.shape
# #             self.input_size = PD
# #             self.feature_names = [f"patch_{i}" for i in range(PD)]

# #         # 7) Return tensors
# #         return {
# #             'markers': patches,  # Now in float32
# #             'labels_strong': torch.from_numpy(labels_seq).long().cuda(),
# #             'batch_idx': idx
# #         }