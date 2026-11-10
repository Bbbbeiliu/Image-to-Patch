import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional

class FSSPrototypeExtractor:

    def __init__(self, encoder):
        self.encoder = encoder

    @torch.no_grad()
    def compute_global_prototype(self, dataloader: DataLoader, device: torch.device, logger=None) -> torch.Tensor:
        was_training = self.encoder.training
        self.encoder.eval()
        feat_sum: Optional[torch.Tensor] = None
        count_sum = 0.0
        try:
            with tqdm(dataloader, desc='Prototype', unit='batch', dynamic_ncols=True) as progress:
                for batch_index, (images, masks, _) in enumerate(progress, 1):
                    images = images.to(device)
                    masks = masks.to(device)
                    feats_list = self.encoder(images)
                    feat = feats_list[-1]
                    masks_ds = F.interpolate(masks, size=feat.shape[2:], mode='bilinear', align_corners=False)
                    masks_bin = (masks_ds > 0.5).float()
                    feat_masked = feat * masks_bin
                    feat_sum_batch = feat_masked.sum(dim=[0, 2, 3])
                    count_batch = masks_bin.sum().item() + 1e-06
                    if feat_sum is None:
                        feat_sum = feat_sum_batch
                    else:
                        feat_sum += feat_sum_batch
                    count_sum += count_batch
                    if logger is not None and (batch_index % 10 == 0 or batch_index == len(dataloader)):
                        logger.info('Prototype %s/%s batches', batch_index, len(dataloader))
                if feat_sum is None:
                    raise ValueError('Prototype support loader is empty')
                prototype = feat_sum / (count_sum + 1e-06)
                prototype = prototype.view(1, -1, 1, 1)
                return prototype.to(device)
        finally:
            self.encoder.train(was_training)

    def fss_loss_from_prototype(self, model, images: torch.Tensor, masks: torch.Tensor, prototype: torch.Tensor, base_criterion, device: torch.device) -> torch.Tensor:
        images = images.to(device)
        masks = masks.to(device)
        feats_list = model.unet.encoder(images)
        feat_q = feats_list[-1]
        feat_q_norm = F.normalize(feat_q, dim=1)
        proto_norm = F.normalize(prototype, dim=1)
        sim_map = (feat_q_norm * proto_norm).sum(dim=1, keepdim=True)
        sim_up = F.interpolate(sim_map, size=masks.shape[2:], mode='bilinear', align_corners=False)
        fss_loss = base_criterion(sim_up, masks)
        return fss_loss
