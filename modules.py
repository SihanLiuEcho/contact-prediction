from torch import nn
import torch
from scipy.spatial.distance import cdist

def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized
def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)

def get_gt_contact_maps(batch_coords):
    '''
    batch_coords: B x L x 3
    contact_maps：B x L x L
    '''
    batch_size, max_len, _ = batch_coords.shape
    batch_dist = torch.empty(
            (
                batch_size,
                max_len,
                max_len,
            ),
            dtype=torch.float32,
        )
    for i, coord in enumerate(batch_coords):
        dist = cdist(coord, coord, metric='euclidean')
        batch_dist[i,:,:] = torch.from_numpy(dist)
    contact_maps = torch.where(batch_dist < 8., 1, 0).float()
    return contact_maps





