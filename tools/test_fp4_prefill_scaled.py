"""Real-weight prefill arithmetic accuracy, null routes and unchanged decode dispatch.

Run with serving stopped. Checks tiny prefills explicitly: the phase, not row
count, selects arithmetic. The old global scaled arm is the dispatch control;
BF16 dequantized matmuls are the independent accuracy reference.
"""
import argparse
import json
import sys
sys.path[:0] = ['/app', '/app/tools']
import torch
import fp4_moe as K
from test_fp4_moe import load_arena, random_routing


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    args=ap.parse_args()
    arena=load_arena(16,'cuda')
    null=15
    for key in ('w1','w3','w2'):
        getattr(arena,key)[null].zero_()
    for key in ('s1','s3','s2'):
        getattr(arena,key)[null].fill_(127)
    gen=torch.Generator().manual_seed(8383)
    rows=[]
    for n in (1,4,17,128,513,2048):
        x=torch.randn(n,K.DIM,generator=gen).bfloat16().cuda()
        slots,weights=random_routing(n,15,gen,'cuda')
        if n>4:
            slots[:,::2]=null
        kw=dict(out_dtype=torch.float32,slots_repeat=True,null_slot=null)
        if n>10:
            kw.update(routing_ids=slots,routing_slot_map=torch.arange(16,device='cuda',dtype=torch.int32))
        K.DOT_SCALED=K.PREFILL_DOT_SCALED=False
        K.CUDA_DECODE=True
        decode=K.moe_forward(x,slots,weights,arena,prefill=False,**kw)
        K.PREFILL_DOT_SCALED=True
        unchanged=K.moe_forward(x,slots,weights,arena,prefill=False,**kw)
        assert torch.equal(decode,unchanged), ('decode changed',n)
        scaled=K.moe_forward(x,slots,weights,arena,prefill=True,**kw)
        K.PREFILL_DOT_SCALED=False
        K.DOT_SCALED=True
        global_control=K.moe_forward(x,slots,weights,arena,**kw)
        assert torch.equal(scaled,global_control), ('wrong dispatch',n)
        ref=K.moe_forward_reference(x,slots,weights,arena,out_dtype=torch.float32)
        rel=float((scaled-ref).norm()/ref.norm())
        assert torch.isfinite(scaled).all() and rel<1e-3, (n,rel)
        K.DOT_SCALED=False
        K.PREFILL_DOT_SCALED=True
        zeros=K.moe_forward(x,torch.full_like(slots,null),weights,arena,prefill=True,
                           out_dtype=torch.float32,slots_repeat=True,null_slot=null)
        assert torch.count_nonzero(zeros)==0
        row=dict(tokens=n,relative_l2=rel,max_abs=float((scaled-ref).abs().max()),
                 decode_exact=True,scaled_dispatch_exact=True,null_zero=True)
        rows.append(row)
        print(json.dumps(row),flush=True)
    with open(args.out,'w') as f:
        json.dump(rows,f,indent=2)


if __name__=='__main__':
    with torch.inference_mode():
        main()
