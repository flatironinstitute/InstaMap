"""Base sensor definitions
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

import instamap.nn.affine

from ..electron_image import ImageConfig
from instamap.simulator.scattering import filters
from instamap.simulator import ctf
import instamap.nn.fft
from instamap.simulator.optics import OpticsShotInfo


class _SafeLog(torch.autograd.Function):
    '''Numerically stable log function that avoids infinite gradients at log(0) by clamping them to at most 1 / finfo.eps
    
    Notes:
    -----
    https://github.com/pyro-ppl/pyro/blob/1a11185ce54a2391348bec5919f3330d957d2f98/pyro/ops/special.py#L15C1-L24C60
    '''
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.log()

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / x.clamp(min=torch.finfo(x.dtype).eps)
    

def safe_log(x):
    """
    Like :func:`torch.log` but avoids infinite gradients at log(0)
    by clamping them to at most ``1 / finfo.eps``.
    """
    return _SafeLog.apply(x)

    
class BioemSensor(torch.nn.Module):
    """
    Equations 4 and 10 on p. 6 of SI in 10.1016/j.jsb.2013.10.006
    observe ~ N*simulate + mu
    with flat uniform (flat) prior on N and mu (method="N-mu")
    with saddle point approx of lambda (method="saddle-approx")
    
    Invariant to sign of simulated and observed (each can arbitrarily change sign and does not affect loss)

    Notes:
    -----
    numerical issues when reconstruction with empirical data using 
        N-mu: nans in up and down after 631/4750 iterations, batch size 2
        saddle-approx: nans in term1 and term2 at iterations 2333/4750, batch size 2
    """

    sigma: torch.Tensor

    def __init__(self, image: ImageConfig, 
                 sigma: float, 
                 N_hi: float = 1.0,
                 N_lo: float = 0.1,
                 mu_hi: float = +10.0,
                 mu_lo: float = -10.0,
                 mask_radius: Optional[float] = None, 
                 method: str = 'saddle-approx'):
        super().__init__()

        self.register_buffer('sigma', torch.tensor(sigma))
        self.register_buffer('N_hi', torch.tensor(N_hi))
        self.register_buffer('N_lo', torch.tensor(N_lo))
        self.register_buffer('mu_hi', torch.tensor(mu_hi))
        self.register_buffer('mu_lo', torch.tensor(mu_lo))

        self.mask_radius = mask_radius
        self.method = method

        if mask_radius is not None:
            self.register_buffer(
                'mask',
                instamap.nn.affine.make_circular_mask(
                    (image.height, image.width), self.mask_radius
                )
            )
        else:
            self.mask = None

    def likelihood(
        self,
        simulated: torch.Tensor,
        observed: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ):
        scale = torch.where(self.sigma > 0, 0.5 / self.sigma.square(), torch.ones_like(self.sigma))

        if self.mask is not None:
            observed = observed * self.mask
            simulated = simulated * self.mask

        eps = torch.finfo(torch.float32).eps

        ccc = simulated.pow(2).sum(dim=(-1,-2))
        if torch.isclose(ccc,torch.zeros_like(ccc)).any():
            print('WARNING: simulator all zeros, so ccc too close to zero. Injecting noise to avoid nans.')
            noise_level = (2*scale).sqrt().pow(-1)
            noise = noise_level*torch.randn(simulated.shape, generator=generator, device=simulated.device, dtype=simulated.dtype)
            simulated = torch.where(ccc.reshape(-1,1,1)==0, simulated + noise, simulated)
            ccc = simulated.pow(2).sum(dim=(-1,-2))

        co = observed.sum(dim=(-1,-2))
        cc = simulated.sum(dim=(-1,-2))
        coo = observed.pow(2).sum(dim=(-1,-2))
        coc = (observed * simulated).sum(dim=(-1,-2))
        
        n_pix = observed.shape[-1] * observed.shape[-2]

        if self.method == 'N-mu':
            up = (n_pix*(ccc*coo-coc*coc) + 2*co*coc*cc -ccc*co*co -coo*cc*cc)
            down = (n_pix*ccc-cc*cc)
            up_over_down = torch.where(torch.logical_and(up==0,down==0), 1,up/down) 
            neg_log_prob = scale*up_over_down + 0.5*safe_log(down.clamp(min=eps)) + (2-n_pix)*safe_log(scale*2)
            assert not neg_log_prob.isnan().any(), 'TODO: numerically stabilize... up={}|down={}'.format(up,down)

        elif self.method == 'saddle-approx':
            term1 = n_pix*(ccc*coo-coc*coc) + 2*co*coc*cc - ccc*co*co - coo*cc*cc
            term2 = (n_pix-2)*(n_pix*ccc-cc*cc)
            neg_log_prob = -(1.5-n_pix/2)*safe_log(term1.clamp(min=eps)) -(n_pix/2-2)*safe_log(term2.clamp(min=eps))
            assert not neg_log_prob.isnan().any(), 'TODO: numerically stabilize... term1={}|term2={}'.format(term1,term2)

        elif self.method == 'N-mu-gaussian-prior-N':

            a = -n_pix*scale

            a2 = (cc*cc/n_pix-ccc)*scale
            b2 = (coc-cc*co/n_pix)*scale
            c2 = (co*co/n_pix - coo) * scale

            lambda_N = 100
            mu_N = 1
            a3 = -1/(2*lambda_N*lambda_N)
            b3 = mu_N / (lambda_N*lambda_N)
            c3 = -mu_N*mu_N/(2*lambda_N*lambda_N)

            neg_log_prob = 0.5*safe_log(-a2-a3) + 0.5*safe_log(-a) + (b2+b3)**2/(4*(a2+a3)) - (c2+c3) + math.log(lambda_N) 


        else:
            raise NotImplementedError("choose a method")

        do_prior = False
        if do_prior:
            beta = 0
            neg_log_prob_prior = (ccc.sqrt() - 1).pow(2) 
            neg_log_prob += beta*n_pix*neg_log_prob_prior
        
        neg_log_prob /= n_pix


        likelihood_scale = simulated.new_tensor(n_pix)

        return neg_log_prob, {'likelihood_scale': likelihood_scale, 'neg_log_prob': neg_log_prob, 'image_loss': neg_log_prob}


    def sample(self, simulated: torch.Tensor, generator: Optional[torch.Generator] = None):
        N = self.N_lo + (self.N_hi - self.N_lo)*torch.rand(simulated.shape[0], generator=generator, device=simulated.device, dtype=simulated.dtype).reshape(-1,1,1)
        mu =self.mu_lo + (self.mu_hi - self.mu_lo)*torch.rand(simulated.shape[0], generator=generator, device=simulated.device, dtype=simulated.dtype).reshape(-1,1,1)
        noise = torch.randn(
            simulated.shape, generator=generator, device=simulated.device, dtype=simulated.dtype
        )
        return N*simulated + noise.mul_(self.sigma) + mu, {}

    def forward(
        self,
        shot_info: Dict[str, torch.Tensor],
        simulated: torch.Tensor,
        observed: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if observed is None:
            return self.sample(simulated, generator=generator)
        else:
            return self.likelihood(simulated, observed, generator=generator)



class CorrelationSensor(torch.nn.Module):
    """Basic sensor with gaussian noise model, but only using the (pre-normalized) correlation term.

    This sensor computes the likelihood of the observed image given a gaussian
    noise model (only correlation term) with fixed variance.
    """

    sigma: torch.Tensor

    def __init__(self, image: ImageConfig, sigma: float, mask_radius: Optional[float] = None):
        super().__init__()

        self.register_buffer('sigma', torch.tensor(sigma))
        self.mask_radius = mask_radius

        if mask_radius is not None:
            self.register_buffer(
                'mask',
                instamap.nn.affine.make_circular_mask(
                    (image.height, image.width), self.mask_radius
                )
            )
        else:
            self.mask = None

    def likelihood(
        self,
        simulated: torch.Tensor,
        observed: torch.Tensor,
        generator: Optional[torch.Generator] = None
    ):
        scale = torch.where(self.sigma > 0, 0.5 / self.sigma.square(), torch.ones_like(self.sigma))
        
        def whiten(arr):
            return (arr - arr.mean(dim=(-1,-2), keepdim=True)) / arr.std(dim=(-1,-2), keepdim=True)

        corr = -2*(whiten(simulated)*whiten(observed))

        if self.mask is not None:
            corr = corr * self.mask

        corr = corr.mean(dim=(-1, -2))
        
        likelihood_scale = corr.new_tensor(simulated.shape[-1] * simulated.shape[-2])

        return corr.mul(scale), {'loss_corr': corr, 'likelihood_scale': likelihood_scale, 'image_loss': corr}

    def sample(self, simulated: torch.Tensor, generator: Optional[torch.Generator] = None):
        noise = torch.randn(
            simulated.shape, generator=generator, device=simulated.device, dtype=simulated.dtype
        )
        return simulated + noise.mul_(self.sigma), {}

    def forward(
        self,
        shot_info: Dict[str, torch.Tensor],
        simulated: torch.Tensor,
        observed: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if observed is None:
            return self.sample(simulated, generator=generator)
        else:
            return self.likelihood(simulated, observed, generator=generator)
        


class HalfCorrelationSensor(torch.nn.Module):
    """Basic sensor with gaussian noise model, but only using the correlation and simulation norm terms.

    This sensor computes the likelihood of the observed image given a gaussian
    noise model (only correlation and simulation norm terms) with fixed variance.
    """

    sigma: torch.Tensor

    def __init__(self, image: ImageConfig, sigma: float, mask_radius: Optional[float] = None):
        super().__init__()

        self.register_buffer('sigma', torch.tensor(sigma))
        self.mask_radius = mask_radius

        if mask_radius is not None:
            self.register_buffer(
                'mask',
                instamap.nn.affine.make_circular_mask(
                    (image.height, image.width), self.mask_radius
                )
            )
        else:
            self.mask = None

    def likelihood(
        self,
        simulated: torch.Tensor,
        observed: torch.Tensor,
        generator: Optional[torch.Generator] = None
    ):
        scale = torch.where(self.sigma > 0, 0.5 / self.sigma.square(), torch.ones_like(self.sigma))
        corr = -2*(simulated*observed)
        norm_protect_inf = simulated.pow(2)

        if self.mask is not None:
            corr = corr * self.mask
            norm_protect_inf = norm_protect_inf * self.mask

        safe_corr = corr.sum(dim=(-1, -2)) + norm_protect_inf.sum(dim=(-1, -2))
        
        likelihood_scale = safe_corr.new_tensor(simulated.shape[-1] * simulated.shape[-2])

        return safe_corr.mul(scale), {'loss_safe_corr': safe_corr, 'likelihood_scale': likelihood_scale, 'image_loss': safe_corr}

    def sample(self, simulated: torch.Tensor, generator: Optional[torch.Generator] = None):
        noise = torch.randn(
            simulated.shape, generator=generator, device=simulated.device, dtype=simulated.dtype
        )
        return simulated + noise.mul_(self.sigma), {}

    def forward(
        self,
        shot_info: Dict[str, torch.Tensor],
        simulated: torch.Tensor,
        observed: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if observed is None:
            return self.sample(simulated, generator=generator)
        else:
            return self.likelihood(simulated, observed, generator=generator)
        

class GaussianSensor(torch.nn.Module):
    """Basic sensor with gaussian noise model.

    This sensor computes the likelihood of the observed image given a gaussian
    noise model with fixed variance.
    """

    sigma: torch.Tensor

    def __init__(self, image: ImageConfig, sigma: float, mask_radius: Optional[float] = None):
        super().__init__()

        self.register_buffer('sigma', torch.tensor(sigma))
        self.mask_radius = mask_radius

        if mask_radius is not None:
            self.register_buffer(
                'mask',
                instamap.nn.affine.make_circular_mask(
                    (image.height, image.width), self.mask_radius
                )
            )
        else:
            self.mask = None

    def likelihood(
        self,
        simulated: torch.Tensor,
        observed: torch.Tensor,
        generator: Optional[torch.Generator] = None
    ):
        scale = torch.where(self.sigma > 0, 0.5 / self.sigma.square(), torch.ones_like(self.sigma))
        square_diff = (simulated - observed).square_()

        if self.mask is not None:
            square_diff = square_diff * self.mask

        mse = square_diff.mean(dim=(-1, -2))

        likelihood_scale = mse.new_tensor(simulated.shape[-1] * simulated.shape[-2])

        likelihood = mse.mul(scale)

        return likelihood, {'loss_mse': mse, 'likelihood_scale': likelihood_scale, 'image_loss': likelihood}

    def sample(self, simulated: torch.Tensor, generator: Optional[torch.Generator] = None):
        noise = torch.randn(
            simulated.shape, generator=generator, device=simulated.device, dtype=simulated.dtype
        )
        return simulated + noise.mul_(self.sigma), {}

    def forward(
        self,
        shot_info: Dict[str, torch.Tensor],
        simulated: torch.Tensor,
        observed: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if observed is None:
            return self.sample(simulated, generator=generator)
        else:
            return self.likelihood(simulated, observed, generator=generator)


class NullSensor(torch.nn.Module):
    """Null sensor model, propagates image when sampling, and does not support likelihood.
    """
    def forward(
        self,
        shot_info: Dict[str, torch.Tensor],
        simulated: torch.Tensor,
        observed: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None
    ):
        if observed is not None:
            raise ValueError('NullSensor does not support observed images')
        return simulated, {}
