import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    def __init__(self, args, dim):
        super().__init__()
        self.rank = args.rank
        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        self.dropepoch=args.dropepoch
        self.eba_strategy = getattr(args, "eba_strategy", "joint")
        self.ce = nn.CrossEntropyLoss(ignore_index=0)

    def _ensure_non_empty_mask(self, mask, scores):
        if torch.any(mask):
            return mask
        fallback = torch.zeros_like(mask)
        fallback[torch.argmax(scores)] = True
        return fallback

    def forward(self, text_features, image_features,local,mlm_scores,mlm_labels,epoch,threshold, threshold_local=None, logit_scale=2.659):

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        device = image_features.device
        sims = image_features @ text_features.T
        threshold_global = threshold
        if threshold_local is None or self.eba_strategy == "joint":
            threshold_local = threshold_global

        # calculated ground-truth and cache if enabled
        # num_logits = logits_per_image.shape[0]  
        num_logits = sims.shape[0]  
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            self.labels[device] = labels
            self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        if epoch<self.dropepoch:
            logits_per_image = logit_scale * sims
            logits_per_text = logit_scale * sims.T
            local1 = local
            local2 = local.T
            global_labels = labels
            local_labels = labels
        else:
            global_pos_sim = torch.diag(sims)
            global_mask = self._ensure_non_empty_mask(global_pos_sim > threshold_global, global_pos_sim)
            if self.eba_strategy == "split":
                local_pos_sim = torch.diag(local)
                local_mask = self._ensure_non_empty_mask(local_pos_sim > threshold_local, local_pos_sim)
            else:
                local_mask = global_mask

            resize_image = sims[global_mask,:]
            resize_text = sims.T[global_mask,:]
            logits_per_image = logit_scale * resize_image
            logits_per_text = logit_scale * resize_text 
            local1 = local[local_mask,:]
            local2 = local.T[local_mask,:]
            global_labels = labels[global_mask]
            local_labels = labels[local_mask]

        # total_loss = (
        #     F.cross_entropy(logits_per_image, labels) +
        #     F.cross_entropy(logits_per_text, labels)
        #     ) 
        total_loss = (
            F.cross_entropy(logits_per_image, global_labels) +
            F.cross_entropy(logits_per_text, global_labels)
            ) 
        localloss2 = F.cross_entropy(local1, local_labels) + F.cross_entropy(local2, local_labels)

        mlmloss=self.ce(mlm_scores, mlm_labels)
        return total_loss+localloss2+0.5*mlmloss
