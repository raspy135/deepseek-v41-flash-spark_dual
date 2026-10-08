"""One graph-safe Markov latency screen; synthetic logits are NOT an acceptance test."""
import json,os,sys,statistics
sys.path[:0]=['/app','/app/tools']
import torch
from safetensors import safe_open
import v41_ref as R
from engine.draft_markov import shortlist_greedy
R.MM_TILE=16
root=os.environ['MODEL_DIR'];index=json.load(open(root+'/model.safetensors.index.json'))['weight_map']
def load(name):
    key='mtp.2.markov_head.'+name+'.weight'
    with safe_open(root+'/'+index[key],framework='pt',device='cpu') as f:
        return f.get_tensor(key).to('cuda',torch.bfloat16)
embed,head=load('embed'),load('head')
torch.manual_seed(71)
logits=torch.randn(5,head.shape[0],device='cuda')*3
anchor=torch.tensor(1234,device='cuda',dtype=torch.int64)
def full():
    prev=anchor.reshape(1);out=[]
    for i in range(5):
        prev=(logits[i]+R.dense(embed[prev],head).float()[0]).argmax().view(1)
        out.append(prev)
    return torch.cat(out)
def limited():return shortlist_greedy(logits,embed,head,anchor,128,R.dense)[0]
report={'note':__doc__}
for name,fn in [('full',full),('shortlist128',limited)]:
    eager=fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=fn()
    times=[]
    for _ in range(5):
        g.replay();a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
        a.record()
        for _ in range(10):g.replay()
        b.record();b.synchronize();times.append(a.elapsed_time(b)/10)
    assert torch.equal(eager,out),'eager/graph proposal mismatch'
    report[name]={'ms':statistics.median(times),'proposal':out.tolist()}
print(json.dumps(report),flush=True)
