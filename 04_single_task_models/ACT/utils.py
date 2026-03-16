import os
import h5py
import math
import torch
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader


class EpisodicDataset(Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, max_episode_len):
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.max_episode_len = max_episode_len

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')

        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']

            qpos = root['/observations/qpos'][()]      # [T, state_dim]
            qvel = root['/observations/qvel'][()]      # [T, state_dim]
            action = root['/action'][()]               # [T, action_dim]

            episode_len = action.shape[0]

            # Sample one timestep as observation anchor
            start_ts = np.random.choice(episode_len)

            qpos_t = qpos[start_ts]
            qvel_t = qvel[start_ts]

            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]

            # Use future action sequence starting at start_ts
            action_seq = action[start_ts:]   # [T-start_ts, action_dim]

        # normalize qpos
        qpos_t = (qpos_t - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        # normalize action sequence
        action_seq = (action_seq - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        # images: stack into [num_cams, C, H, W]
        all_cam_images = []
        for cam_name in self.camera_names:
            curr_image = image_dict[cam_name]   # [H, W, 3]
            curr_image = np.transpose(curr_image, (2, 0, 1)) / 255.0
            all_cam_images.append(curr_image)
        image_data = np.stack(all_cam_images, axis=0).astype(np.float32)

        # pad action sequence to fixed length = max_episode_len
        action_dim = action_seq.shape[1]
        padded_action = np.zeros((self.max_episode_len, action_dim), dtype=np.float32)
        is_pad = np.ones(self.max_episode_len, dtype=np.float32)

        action_len = min(len(action_seq), self.max_episode_len)
        padded_action[:action_len] = action_seq[:action_len]
        is_pad[:action_len] = 0.0

        qpos_data = torch.from_numpy(qpos_t).float()
        image_data = torch.from_numpy(image_data).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    episode_lens = []

    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]   # [T, state_dim]
            action = root['/action'][()]            # [T, action_dim]

        qpos = torch.from_numpy(qpos).float()
        action = torch.from_numpy(action).float()

        all_qpos_data.append(qpos)
        all_action_data.append(action)
        episode_lens.append(len(action))

    # variable-length episodes: concatenate over time
    all_qpos_data = torch.cat(all_qpos_data, dim=0)       # [sum(T), state_dim]
    all_action_data = torch.cat(all_action_data, dim=0)   # [sum(T), action_dim]

    qpos_mean = all_qpos_data.mean(dim=0)
    qpos_std = all_qpos_data.std(dim=0)
    action_mean = all_action_data.mean(dim=0)
    action_std = all_action_data.std(dim=0)

    # avoid divide-by-zero
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)
    action_std = torch.clip(action_std, 1e-2, np.inf)

    stats = {
        'qpos_mean': qpos_mean.numpy(),
        'qpos_std': qpos_std.numpy(),
        'action_mean': action_mean.numpy(),
        'action_std': action_std.numpy(),
    }

    max_episode_len = max(episode_lens)
    return stats, max_episode_len


def find_all_hdf5(dataset_dir):
    files = []
    i = 0
    while True:
        path = os.path.join(dataset_dir, f'episode_{i}.hdf5')
        if not os.path.exists(path):
            break
        files.append(path)
        i += 1
    return files


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')

    # collect stats
    norm_stats, max_episode_len = get_norm_stats(dataset_dir, num_episodes)

    # split train / val
    shuffled_indices = np.arange(num_episodes)
    np.random.shuffle(shuffled_indices)

    train_ratio = 0.8
    train_episode_len = math.ceil(train_ratio * num_episodes)

    train_indices = shuffled_indices[:train_episode_len]
    val_indices = shuffled_indices[train_episode_len:]

    train_dataset = EpisodicDataset(
        train_indices,
        dataset_dir,
        camera_names,
        norm_stats,
        max_episode_len,
    )
    val_dataset = EpisodicDataset(
        val_indices,
        dataset_dir,
        camera_names,
        norm_stats,
        max_episode_len,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )

    return train_dataloader, val_dataloader, norm_stats, max_episode_len


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    peg_position = np.array([0.1, 0.5, 0.05])
    peg_quat = np.array([1, 0, 0, 0])

    socket_position = np.array([-0.1, 0.5, 0.05])
    socket_quat = np.array([1, 0, 0, 0])

    return (
        np.concatenate([peg_position, peg_quat]),
        np.concatenate([socket_position, socket_quat]),
    )


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)

    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items

    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)