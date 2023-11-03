import torch
import torch.nn as nn
from taming.modules.losses.vqperceptual import *


class LPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        disc_num_layers=3,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        disc_loss="hinge",
        kl_weight=1.0,
        perceptual_weight=1.0,
        use_actnorm=False,
    ):
        super().__init__()

        assert disc_loss in ["hinge", "vanilla"]
        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight

        # self.kl_weight = kl_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.logvar = nn.Parameter(torch.ones(size=()) * 0)

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight *= self.discriminator_weight
        return d_weight

    def forward(
        self,
        inputs,
        reconstructions,
        posteriors,
        optimizer_idx,
        global_step,
        last_layer,
        cond,
    ):
        rec_loss = torch.abs(cond.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(
                cond.contiguous(), reconstructions.contiguous()
            )
            rec_loss += self.perceptual_weight * p_loss
        nll_loss = torch.mean(rec_loss)

        if optimizer_idx == 0:
            logits_fake = self.discriminator(reconstructions.contiguous())
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            loss = nll_loss + d_weight * disc_factor * g_loss

            log = {
                "total_loss": loss.clone().detach().mean(),
                "rec_loss": rec_loss.detach().mean(),
                "d_weight": d_weight.detach(),
                "disc_factor": torch.tensor(disc_factor),
                "g_loss": g_loss.detach().mean(),
            }

            return loss, log

        if optimizer_idx == 1:
            logits_real = self.discriminator(cond.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "disc_loss": d_loss.clone().detach().mean(),
                "logits_real": logits_real.detach().mean(),
                "logits_fake": logits_fake.detach().mean(),
            }

            return d_loss, log