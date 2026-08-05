import torch
import stable_worldmodel as swm

m = swm.wm.utils.load_pretrained('/workspace/ckpts/dinowm_noprop_cube').eval().cuda()
print('LOADED:', type(m).__name__)
for res in (196, 224):
    info = {'pixels': torch.randn(2, 3, 3, res, res).cuda(),
            'action': torch.randn(2, 3, 25).cuda()}
    try:
        with torch.no_grad():
            out = m.encode(info)
            z = m.predictor(out['emb'].flatten(1, 2))
        emb = tuple(out['emb'].shape)
        print('res %d: emb %s -> predictor %s  OK' % (res, emb, tuple(z.shape)))
    except Exception as e:
        print('res %d: FAILED %s: %s' % (res, type(e).__name__, str(e)[:140]))
