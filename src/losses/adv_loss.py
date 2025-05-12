from typing import Callable
import contextlib

import torch
from torch import nn
import torch.nn.functional as F

from .dice_loss import DiceLoss


def _l2_normalize(d: torch.Tensor) -> torch.Tensor:
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, "track_running_stats"):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


class VAT2d(nn.Module):

    def __init__(
        self,
        xi: float = 10.0,
        epi: float = 6.0,
        ip: int = 1,
        loss_cls: Callable = DiceLoss,
        loss_kwargs: dict = {"num_classes": 3, "do_bg": True},
    ):
        super(VAT2d, self).__init__()
        self.xi = xi
        self.epi = epi
        self.ip = ip
        self.loss = loss_cls(**loss_kwargs)

    def forward(
        self,
        model,
        x,
        image_size,
        multimask_output=True,
        prompt_idx=-1,
        promptmode=None,
        image_embeddings=None,
        outputs=None,
    ):
        with torch.no_grad():
            if outputs is None:
                outputs = model(
                    x,
                    multimask_output,
                    image_size,
                    prompt_idx,
                    promptmode,
                    image_embeddings,
                )
            pred = torch.zeros(1, device=model.device)
            for m in outputs["low_res_logits"]:
                pred = pred + m.softmax(1)
            pred /= len(outputs["low_res_logits"])

        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_(True)

                r_hat = d * self.xi

                # If image_embeddings is provided, we apply the noise to it
                # Therefore, the main input is image_embeddings and not the 
                # original image.
                if image_embeddings is not None:
                    image_embeddings_hat = image_embeddings + r_hat
                else:
                    image_embeddings_hat = None

                outputs_hat = model(
                    x + r_hat,
                    multimask_output,
                    image_size,
                    prompt_idx,
                    promptmode,
                    image_embeddings_hat,
                )
                pred_hat = torch.zeros(1, device=model.device)
                for m in outputs_hat["low_res_logits"]:
                    pred_hat = pred_hat + m.softmax(1)
                pred_hat /= len(outputs_hat["low_res_logits"])

                adv_distance = self.loss(pred_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            r_adv = d * self.epi
            # If image_embeddings is provided, we apply the noise to it
            # Therefore, the main input is image_embeddings and not the 
            # original image.
            if image_embeddings is not None:
                image_embeddings_hat = image_embeddings + r_adv
            else:
                image_embeddings_hat = None

            outputs_hat = model(
                x + r_adv,
                multimask_output,
                image_size,
                prompt_idx,
                promptmode,
                image_embeddings_hat,
            )
            pred_hat = torch.zeros(1, device=model.device)
            for m in outputs_hat["low_res_logits"]:
                pred_hat = pred_hat + m.softmax(1)
            pred_hat /= len(outputs_hat["low_res_logits"])
            lds = self.loss(pred_hat, pred)

        return lds
