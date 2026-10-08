"""One32K-group gate: adversarial finite math first; overflow reported separately."""
import argparse
import json
import hashlib
import statistics
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from fp4_group_tc import project_group, reference_group, project_group_out, reference_group_out


def packed(codes):
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def compare(name, x, weights, *, single_products):
    native = reference_group(x, weights)
    actual = project_group(x, weights, single_products=single_products)
    torch.cuda.synchronize()
    mismatch = actual.view(torch.int32) != native.view(torch.int32)
    coords = mismatch.nonzero().cpu().tolist()
    row = {'name': name, 'shape': list(native.shape), 'elements': native.numel(),
        'changed_bits': len(coords), 'native_finite': bool(torch.isfinite(native).all()),
        'converted_activations_finite': bool(torch.isfinite(x.half()).all()),
        'first_changes': [{'coordinate': c, 'native': float(native[tuple(c)]),
             'candidate': float(actual[tuple(c)]),
             'native_bits': int(native.view(torch.int32)[tuple(c)]),
             'candidate_bits': int(actual.view(torch.int32)[tuple(c)])} for c in coords[:16]]}
    print(json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--single-products', action='store_true')
    ap.add_argument('--timing', action='store_true', help='tiny hot group helper timing after qualification')
    args = ap.parse_args()
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {'device': torch.cuda.get_device_name(), 'finite_cases': [], 'overflow': [],
              'single_products': args.single_products,
              'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (Path(__file__), Path(__file__).with_name('fp4_group_tc.py'),
                               Path(__file__).with_name('fp4_moe_cuda.cu'))},
              'completed': False, 'exact_finite': False, 'passed': False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    def save(): args.out.write_text(json.dumps(report, indent=2)+'\n')
    def run(name, x, weights):
        return compare(name, x, weights, single_products=args.single_products)
    save()

    # Every BF16 code that remains finite after conversion to FP16, packed into
    # contiguous32-value groups. This includes both zeros and underflow boundaries.
    bits = torch.arange(65536, device='cuda', dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16)
    values = values[torch.isfinite(values.half())]
    assert values.numel()%32 == 0
    x = values.reshape(-1, 1, 32).contiguous()
    c = torch.arange(16, device='cuda', dtype=torch.uint8)[None, :, None]
    weights = packed(c.expand(x.shape[0],16,32))
    report['finite_cases'].append(run('all_finite_converted_bf16_codes_all16_weights',x,weights));save()

    # All256 packed byte patterns, each exercised by sparse impulses and opposing
    # dynamic ranges. Include representable FP16 subnormals and signed zero.
    vals = [0., -0., 2**-25, -(2**-25), 2**-24, -(2**-24),
        2**-23, -(2**-23), 2**-14, -(2**-14), 1., -1., 65280., -65280.]
    seeds = torch.tensor(vals,device='cuda',dtype=torch.bfloat16)
    x = seeds.repeat(5)[:64].reshape(1,2,32).repeat(8,1,1).contiguous()
    for i in range(8):
        x[i] = x[i].roll(i, -1)
    weights = torch.arange(256,device='cuda',dtype=torch.uint8)[None,:,None].expand(8,256,16).contiguous()
    report['finite_cases'].append(run('all256_byte_patterns_boundaries_cancellation',x,weights));save()

    torch.manual_seed(107)
    exponent = torch.randint(-24,16,(96,2,32),device='cuda')
    mantissa = torch.randint(128,256,(96,2,32),device='cuda').float()/128
    sign = torch.where(torch.rand(96,2,32,device='cuda')>.5,1.,-1.)
    x = (sign*mantissa*torch.exp2(exponent.float())).bfloat16()
    codes = torch.randint(0,16,(96,39,32),device='cuda',dtype=torch.uint8)
    report['finite_cases'].append(run('mixed_codes_signed_logrange_n39_m2',x,packed(codes)));save()

    # An explicit old eight-leaf cancellation tree: middle-lane signs matter.
    x = torch.tensor([65280.,1.,-65280.,1.]*8,device='cuda',dtype=torch.bfloat16)
    x = x.reshape(1,1,32).repeat(16,1,1)
    codes = torch.ones(16,32,32,device='cuda',dtype=torch.uint8)*2
    for i in range(16): codes[i] = codes[i].roll(i,-1)
    report['finite_cases'].append(run('eight_leaf_tree_cancellation',x,packed(codes)));save()

    huge = torch.tensor([2**16,-(2**16),0.,-0.]*8,device='cuda',dtype=torch.bfloat16)
    x = huge.reshape(1,1,32).repeat(2,1,1)
    codes = torch.arange(16,device='cuda',dtype=torch.uint8)[None,:,None].expand(2,16,32)
    report['overflow'].append(run('bf16_finite_half_overflow_report_only',x,packed(codes)))
    if args.single_products:
        values = torch.tensor([2.**e*m for e in range(-24,4)
            for m in (1.,1.5,1.75,-1.,-1.5,-1.75)],device='cuda',dtype=torch.bfloat16)
        x = torch.zeros((values.numel(),1,32),device='cuda',dtype=torch.bfloat16)
        x[:,0,0] = 1.; x[:,0,2] = values
        weights = torch.full((values.numel(),32,16),0x22,device='cuda',dtype=torch.uint8)
        report['rejected_two_product_isolation'] = compare('two_product_alignment_isolation',
            x,weights,single_products=False)
        report['finite_cases'].append(run('single_product_alignment_isolation',x,weights))
    report['completed'] = True
    report['exact_finite'] = all(row['changed_bits']==0 and row['native_finite']
                                  and row['converted_activations_finite'] for row in report['finite_cases'])
    report['passed'] = report['exact_finite']
    save()
    assert report['exact_finite'], 'sparse virtual-row TC changed meaningful finite native group partials'
    if args.timing:
        # One-group hot arithmetic screen, not a full-K bandwidth/model claim.
        # Both outputs are persistent and captured to remove Python/allocation overhead.
        report['group_timing'] = []
        for members in (1,2):
            x = (torch.randn(256,members,32,device='cuda')*.5).bfloat16()
            weights = torch.randint(0,256,(256,1152,16),device='cuda',dtype=torch.uint8)
            outputs = {arm: torch.empty((256,members,1152),device='cuda') for arm in ('simt','tc')}
            launch = {'simt': lambda: reference_group_out(x,weights,outputs['simt']),
                      'tc': lambda: project_group_out(x,weights,outputs['tc'],single_products=args.single_products)}
            graphs = {}
            for arm, fn in launch.items():
                for _ in range(3): fn()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(32): fn()
                graphs[arm] = graph
            samples = {arm: [] for arm in graphs}
            for quartet in range(6):
                for arm in (('simt','tc','tc','simt') if quartet%2==0 else ('tc','simt','simt','tc')):
                    graphs[arm].replay()
                    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();graphs[arm].replay();end.record();end.synchronize()
                    samples[arm].append(start.elapsed_time(end)*1000/32)
            assert torch.equal(outputs['tc'].view(torch.int32),outputs['simt'].view(torch.int32))
            report['group_timing'].append({'shape': [256,members,1152,32], 'calls_per_graph': 32,
                'quartets': 6, 'hot_payload_bytes': weights.numel()+x.numel()*2,
                'microseconds': {arm: statistics.median(values) for arm,values in samples.items()},
                'samples_microseconds': samples,
                'note': 'One group helper only; inline CUDA SIMT reference repeats activation/weight loads per member.'})
            save()
    print('FP4_GROUP_TC_FINITE_EXACT', flush=True)


if __name__ == '__main__':
    main()
