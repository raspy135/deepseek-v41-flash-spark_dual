"""Measure cuBLAS workspace retention for new versus reused warm-up streams.

Run inside the engine image. No model load; a seconds-long regression probe
for why FastDecoder must keep its capture_warmup_stream across graph rotations.
"""
import json
import torch


def main():
    x = torch.randn(16, 5120, device='cuda')
    weight = torch.randn(5120, 384, device='cuda')
    output = torch.empty(16, 384, device='cuda')
    streams, fresh, reused = [], [], []
    for _ in range(6):
        stream = torch.cuda.Stream()
        streams.append(stream)
        with torch.cuda.stream(stream):
            torch.mm(x, weight, out=output)
        stream.synchronize()
        fresh.append(torch.cuda.memory_allocated())
    for _ in range(6):
        with torch.cuda.stream(stream):
            torch.mm(x, weight, out=output)
        stream.synchronize()
        reused.append(torch.cuda.memory_allocated())
    assert len(set(reused)) == 1, 'reusing a warmed stream grew allocated memory'
    print(json.dumps(dict(torch=torch.__version__, new_stream_bytes=fresh,
                          reused_stream_bytes=reused), indent=2))


if __name__ == '__main__':
    main()
