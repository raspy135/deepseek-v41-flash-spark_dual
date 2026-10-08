"""UNRUN CUDA DRIVER: CPU geometry first, bounded bit gate and optional timing.

CPU: python3 tools/test_attention_softmax_exact.py
CUDA: python tools/test_attention_softmax_exact.py --cuda --timing --out proof.json
The default CUDA screen covers aligned serving widths only; --tails adds the
experimental alignment path.  Peak live tensors are below 64 MiB; the separate
CUDA context itself is larger.  No model or checkpoint is loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attention_softmax_exact import attn_probs, lane_streams, reduction_config, sum_row_cpu, sum_rows


class GeometryCPU(unittest.TestCase):
    def test_serving_geometry(self):
        for rows in (32, 64, 128, 192):
            for n in (128, 640, 1152, 3200):
                cfg = reduction_config(n, rows)
                self.assertEqual((cfg.width, cfg.height, cfg.vectorized, cfg.split_y), (32,16,True,False))

    def test_every_index_once_including_header_tail(self):
        for rows in (1, 8, 32, 128):
            for n in (1, 3, 17, 31, 32, 63, 127, 128, 129, 133, 640, 1152, 3200):
                cfg = reduction_config(n, rows)
                if cfg.split_y and cfg.height > 1:
                    continue
                for row in range(4):
                    visited = [i for lane in lane_streams(cfg,row) for acc in lane for i in acc]
                    self.assertEqual(sorted(visited), list(range(n)), (cfg,row))

    def test_adversarial_grouping_is_not_sequential_sum(self):
        p = [0.] * 128
        p[0], p[1], p[2], p[3] = 2.**25, 1., -(2.**25), 1.
        # Four float4 accumulator streams are folded left-to-right per lane.
        self.assertEqual(sum_row_cpu(p,32), 1.)
        p[1], p[64] = 0., 1.
        # The descending cross-lane tree preserves both small contributions.
        self.assertEqual(sum_row_cpu(p,32), 2.)

    def test_alignment_headers_are_in_accumulator_zero(self):
        streams = lane_streams(reduction_config(133,32),1)
        self.assertEqual(streams[1][0][0],0)
        self.assertEqual(streams[2][0][0],1)
        self.assertEqual(streams[3][0][0],2)
        self.assertEqual(streams[0][0][0],3)
        self.assertEqual(streams[0][0][-1],131)
        self.assertEqual(streams[1][0][-1],132)


def bit_report(torch, name, actual, reference):
    mismatch = actual.view(torch.int32) != reference.view(torch.int32)
    positions = mismatch.nonzero().cpu().tolist()
    return {'name':name,'elements':actual.numel(),'changed_bits':len(positions),
            'max_abs_finite':float(torch.where(torch.isfinite(actual-reference),
                    (actual-reference).abs(),0.).max()),
            'first_changes':[{'coordinate':p,'reference':float(reference[tuple(p)]),
                    'actual':float(actual[tuple(p)]),
                    'reference_bits':int(reference.view(torch.int32)[tuple(p)]),
                    'actual_bits':int(actual.view(torch.int32)[tuple(p)])} for p in positions[:8]]}


def cuda_gate(args):
    import torch
    import triton
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from engine.decode_lean import LeanOps
    torch.cuda.set_device(0)
    torch.manual_seed(413)
    lean = LeanOps('cuda',16,32)
    report = {'torch':torch.__version__,'triton':triton.__version__,
        'device':torch.cuda.get_device_name(),'completed':False,'passed':False,
        'cases':[],'graphs':[],'timing':[],
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),Path(__file__).with_name('attention_softmax_exact.py'))}}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    def save(): args.out.write_text(json.dumps(report,indent=2)+'\n')
    def append(name, actual, reference):
        result = bit_report(torch,name,actual,reference)
        report['cases'].append(result);save();print(json.dumps(result),flush=True)
        assert result['changed_bits']==0, name+' changed FP32 bits'
    save()

    widths = (128,640,1152,3200) + ((17,127,129,133) if args.tails else ())
    for n in widths:
        # Isolate reduction from exp/div: wide positive dynamic range, sparse
        # rounding-boundary impulses, cancellation, both zeros, normal/subnormal.
        p = torch.exp2(torch.randint(-149,8,(32,n),device='cuda').float())
        p[0].zero_();p[1].fill_(1.)
        p[2].zero_();p[2,:4]=torch.tensor([2.**25,1.,-(2.**25),1.],device='cuda')
        p[3].fill_(2.**-25);p[3,0]=1.
        p[4]=p[4]*torch.where(torch.arange(n,device='cuda')%2==0,1.,-1.)
        p[5].fill_(-0.);p[6].fill_(2.**-149)
        append(f'sum_n{n}',sum_rows(p),p.sum(-1,keepdim=True))
        # Compare scalar CPU derivation for four representative rows too.
        cpu=p[:7].cpu().tolist()
        expected=torch.tensor([sum_row_cpu(row,32,i) for i,row in enumerate(cpu)],
                              device='cuda',dtype=torch.float32)[:,None]
        append(f'cpu_tree_n{n}',sum_rows(p)[:7],expected)

        for t in (1,2,4,6):
            scores=torch.randn((t,32,n),device='cuda')*8
            sink=torch.linspace(-80.,80.,32,device='cuda')
            sink[0],sink[1]=0.,-0.
            mask=torch.rand((t,n),device='cuda')>.2
            # All-masked query, isolated maximum, very flat probabilities,
            # exp underflow boundaries and zero scores coexist in one shape.
            if t>1: mask[0]=False
            scores[:,0]=0.
            scores[:,1]=torch.linspace(-104.,0.,n,device='cuda')/(512**-.5)
            scores[:,2]=-(25.*0.6931471805599453)/(512**-.5)
            scores[:,2,0]=0.
            reference=lean.attn_probs(scores,mask,sink,512**-.5)
            actual,total,mx=attn_probs(scores,mask,sink,512**-.5,debug=True)
            append(f'softmax_t{t}_n{n}',actual,reference)
            from engine.decode_lean import _softmax_pre_kernel
            exp_reference=torch.empty_like(scores)
            max_reference=torch.empty((t,32,1),device='cuda')
            _softmax_pre_kernel[(t*32,)](scores,mask.view(torch.uint8),exp_reference,
                max_reference,32,n,*scores.stride(),512**-.5,
                BN=triton.next_power_of_2(n),num_warps=4,
                enable_fp_fusion=False,enable_reflect_ftz=False)
            append(f'softmax_sum_t{t}_n{n}',total,exp_reference.sum(-1,keepdim=True))
            append(f'softmax_max_t{t}_n{n}',mx,max_reference)

    # Arbitrary score strides and a mask requiring contiguous conversion.
    n=1152;scores=torch.randn((4,n,32),device='cuda').transpose(1,2)
    mask=(torch.rand((n,4),device='cuda')>.1).transpose(0,1)
    sink=torch.randn(32,device='cuda')
    append('strided_scores_mask',attn_probs(scores,mask,sink,.03125),
                                lean.attn_probs(scores,mask,sink,.03125))
    # Replay a single captured candidate while mutating every input in place.
    scores=scores.contiguous();mask=mask.contiguous()
    output=torch.empty_like(scores)
    attn_probs(scores,mask,sink,.03125,out=output)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):attn_probs(scores,mask,sink,.03125,out=output)
    for mutation in range(3):
        scores.normal_();mask.copy_(torch.rand_like(mask,dtype=torch.float32)>.25)
        sink.normal_();graph.replay()
        result=bit_report(torch,f'graph_mutation_{mutation}',output,
                         lean.attn_probs(scores,mask,sink,.03125))
        report['graphs'].append(result);save()
        assert result['changed_bits']==0,'graph replay changed bits'

    report['completed']=True;report['passed']=True
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated();save()
    if args.timing:
        # Tiny graph screen after raw-bit qualification; no model throughput claim.
        for n in (128,1152,3200):
            scores=torch.randn((4,32,n),device='cuda');sink=torch.randn(32,device='cuda')
            mask=torch.ones((4,n),device='cuda',dtype=torch.bool)
            output=torch.empty_like(scores)
            functions=(lambda:lean.attn_probs(scores,mask,sink,512**-.5),
                       lambda:attn_probs(scores,mask,sink,512**-.5,out=output))
            graphs=[]
            for fn in functions:
                fn();torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(32):fn()
                graphs.append(graph)
            samples=[[],[]]
            for _ in range(6):
                for which in (0,1,1,0):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();graphs[which].replay();end.record();end.synchronize()
                    samples[which].append(start.elapsed_time(end)*1000/32)
            row={'n':n,'t':4,'h':32,'baseline_us':statistics.median(samples[0]),
                'fused_us':statistics.median(samples[1]),'samples_us':samples}
            report['timing'].append(row);save();print(json.dumps(row),flush=True)
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated();save()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda',action='store_true')
    parser.add_argument('--tails',action='store_true')
    parser.add_argument('--timing',action='store_true')
    parser.add_argument('--out',type=Path,default=Path('results/attention-softmax-exact/proof.json'))
    args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GeometryCPU))
    if not result.wasSuccessful():raise SystemExit(1)
    if args.cuda:cuda_gate(args)


if __name__=='__main__':main()
