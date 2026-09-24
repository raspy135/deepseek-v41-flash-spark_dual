"""CPU ordering checks; the TP gate validates real CUDA replay and numerics."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class SharedOverlapTests(unittest.TestCase):
    def run_layer(self, enabled, fail=False):
        events = []
        class Tensor:
            def __init__(self, value): self.value = value
            def float(self): return self
            def to(self, dtype): return self
            def record_stream(self, stream): events.append('lifetime')
            def __iadd__(self, rhs):
                events.append('add')
                self.value += rhs.value
                return self
            def copy_(self, rhs): self.value = rhs.value
        class Stream:
            def __init__(self, name): self.name = name
            def wait_stream(self, other): events.append((self.name, 'wait', other.name))
            def __enter__(self): events.append('shared-enter')
            def __exit__(self, *exc): events.append('shared-exit')
        main, side = Stream('main'), Stream('side')
        cuda = SimpleNamespace(current_stream=lambda dev: main, stream=lambda s: s)
        def shared(*args):
            events.append('shared')
            return Tensor(3)
        def routed():
            events.append('routed')
            if fail:
                raise RuntimeError('routed failure')
            return Tensor(7)
        # Execute the actual method without importing CUDA/Triton on a CPU host.
        source = Path(__file__).resolve().parents[1] / 'engine/fastdecode.py'
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FastDecoder')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_layer_b')
        scope = dict(torch=SimpleNamespace(cuda=cuda, bfloat16='bf16'), _HC_OPS=True,
                     R=SimpleNamespace(expert_ffn=shared),
                     _hc_post_fused=lambda out, *rest: out)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
        decoder = SimpleNamespace(a=SimpleNamespace(swiglu_limit=10),
            W=SimpleNamespace(layers=[SimpleNamespace(sh_w1=None, sh_w2=None, sh_w3=None)]),
            shared_stream=side if enabled else None, dev='cuda', y=Tensor(1),
            h=Tensor(0), pre_mix=Tensor(0), ffn_pre=Tensor(2), ffn_post=None,
            ffn_comb=None, _routed_experts=routed)
        if fail:
            with self.assertRaisesRegex(RuntimeError, 'routed failure'):
                scope['_layer_b'](decoder, 0)
        else:
            scope['_layer_b'](decoder, 0)
            self.assertEqual(decoder.h.value, 10)
            self.assertEqual(decoder.pre_mix.value, 2)
        return events

    def test_disabled_preserves_serial_order(self):
        self.assertEqual(self.run_layer(False), ['routed', 'shared', 'add'])

    def test_enabled_forks_and_joins_before_add(self):
        self.assertEqual(self.run_layer(True), [
            ('side', 'wait', 'main'), 'shared-enter', 'shared', 'shared-exit',
            'routed', ('main', 'wait', 'side'), 'lifetime', 'add'])

    def test_failure_still_joins(self):
        events = self.run_layer(True, fail=True)
        self.assertEqual(events[-1], ('main', 'wait', 'side'))
        self.assertNotIn('add', events)


if __name__ == '__main__':
    unittest.main()
