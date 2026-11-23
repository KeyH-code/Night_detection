"""Source-only Fourier appearance replay from the same approved mixed-domain microbatch."""
from __future__ import annotations

import night_focus
import night_rank
from source_contract import ContractError, check_equal


def source_style(images, content, frequency, strength):
    import torch

    if images.ndim!=4 or images.shape[:2]!=(2,3) or not 0<frequency<=.20 or not 0<=strength<=1:
        raise ContractError('Invalid signed frequency patch or paired source/night layout')
    if not strength:
        return images
    with torch.no_grad():
        pixels=images[:,:,content,:].float()
        h,w=pixels.shape[-2:]
        src,night=torch.fft.rfft2(pixels).unbind(0)
        fy=torch.fft.fftfreq(h,device=src.device).abs()
        fx=torch.fft.rfftfreq(w,device=src.device)
        region=(fy[:,None]<=frequency)&(fx[None,:]<=frequency)
        amplitude=src.abs()
        mix=amplitude.lerp(night.abs(),strength)
        gain=torch.where(amplitude>1e-9,mix/amplitude.clamp_min(1e-9),torch.ones_like(amplitude))
        updated=torch.where(region,src*gain,src)
        styled=torch.fft.irfft2(updated,s=(h,w)).clamp_(0,1).to(images.dtype)
        result=images.clone()
        result[0,:,content,:]=styled
    return result


def validate(cfg):
    check_equal('registered frequency style treatment',(2000,200,.05,.8,40),
                tuple(cfg[k] for k in ('day_images','updates','frequency','transfer_strength','max_minutes')))


def fit(torch,cv2,payloads,plan,seed,original,cfg,until,phase):
    sizes={tuple(row[1:3]) for domain in payloads.values() for row in domain.values()}
    check_equal('frozen no-crop geometry for style',({(1920,1080)},640),(sizes,original['size']))
    content=slice(140,500)
    sums=torch.zeros(3,device='cuda',dtype=torch.float32)
    observed=[]

    def transform(images,i):
        if i%2==0:
            return images
        output=source_style(images,content,cfg['frequency'],cfg['transfer_strength'])
        if not observed:
            check_equal('style preserves target tensor and source padding',True,
                        torch.equal(output[1],images[1]) and torch.equal(output[0,:,:140],images[0,:,:140])
                        and torch.equal(output[0,:,500:],images[0,:,500:]))
            check_equal('style changed only a finite source content tensor',True,
                        bool(output.isfinite().all()) and not torch.equal(output[0],images[0]))
        with torch.no_grad():
            sums.add_(torch.stack((images[0,:,content].mean(),images[1,:,content].mean(),output[0,:,content].mean())))
        observed.append(i)
        return output

    path,record=night_focus.fit_native(torch,cv2,payloads,plan,seed,original,cfg,until,phase,transform=transform)
    check_equal('source only half exposure, five per update',list(range(1,2000,2)),observed)
    check_equal('native BCE and no tail loss intervention',[],record['focus_classes'])
    return path,dict(record,source_style_exposures=len(observed),style_interval='odd microbatch, source only',
                     style_content_rows=[140,500],style_frequency=cfg['frequency'],style_strength=cfg['transfer_strength'],
                     mean_pixels_day_night_changed=(sums/len(observed)).cpu().tolist())


if __name__=='__main__':
    night_rank.main(kind='style',fit=fit,validate=validate,description=__doc__)
