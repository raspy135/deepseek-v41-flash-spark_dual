"""Cold graph microbenchmark: emulate 15us RoCE wait, then consume weights.

A screen, not an end-to-end speed claim. All variants must read identical bytes.
"""
import sys,json,statistics
sys.path[:0]=['/app','/app/tools']
import torch,triton
from engine import l2pf
from tools.bench_l2_prefetch import _read, _sleep_ms
from triton.testing import do_bench_cudagraph

cycles=max(1,int(100000*(.015/_sleep_ms(100000,20))))
print("sleep cycles",cycles,flush=True)
side=torch.cuda.Stream()
report=[]
for mb in (2,4,8,12):
 # > L2 between reuse, also verify input weights stay unchanged.
 xs=[torch.randn(mb*1024**2//4,device='cuda') for _ in range(32)]
 outs=[torch.zeros(triton.cdiv(x.numel(),4096),device='cuda') for x in xs]
 refs=[x.clone() for x in xs[:1]]
 def body(mode,pace):
  l2pf.MODE=mode;l2pf.PACE_GBPS=pace
  for x,out in zip(xs,outs):
   if mode!='off':
    side.wait_stream(torch.cuda.current_stream())
    l2pf.touch(side,[x],mb*1024**2)
   torch.cuda._sleep(cycles)
   if mode!='off':torch.cuda.current_stream().wait_stream(side)
   _read[(out.numel(),)](x,out,x.numel(),BLOCK=4096,num_warps=4)
 row={'mb':mb,'us':{}}
 for mode,pace in [('off',0),('touch',0),('bulk',0),('bulk',150),('bulk',100)]:
  print('START',mb,mode,pace,flush=True)
  body(mode,pace)
  torch.cuda.synchronize()
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):body(mode,pace)
  for _ in range(3):g.replay()
  st,en=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  st.record()
  for _ in range(30):g.replay()
  en.record();en.synchronize()
  v=st.elapsed_time(en)*1000/30/len(xs)
  row['us'][f'{mode}-{pace}']=v
  assert torch.equal(xs[0],refs[0])
 report.append(row);print(json.dumps(row),flush=True)
