"""Declared technical perturbations with measured per-record noise power."""

import torch


def corrupt(x,snr_db=12.,dropout=0.):
    """Mix baseline drift, electrode-motion transients and white noise at a fixed SNR."""
    t=torch.linspace(0,1,x.shape[-1],device=x.device)
    phase=torch.rand(*x.shape[:-1],1,device=x.device)*2*torch.pi
    baseline=torch.sin(2*torch.pi*(1+2*torch.rand_like(phase))*t+phase)
    motion=torch.randn_like(x).cumsum(-1)
    motion=motion-motion.mean(-1,keepdim=True)
    motion=motion/motion.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-6)
    noise=.4*baseline+.4*motion+.2*torch.randn_like(x)
    noise=noise-noise.mean((1,2),keepdim=True)
    signal_power=x.square().mean((1,2),keepdim=True).clamp_min(1e-8)
    noise=noise*(signal_power/noise.square().mean((1,2),keepdim=True).clamp_min(1e-8)/10**(snr_db/10)).sqrt()
    damaged=x+noise
    if dropout:
        damaged=damaged.masked_fill(torch.rand(x.shape[0],x.shape[1],1,device=x.device)<dropout,0)
    return damaged
