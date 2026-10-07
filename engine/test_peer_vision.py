import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from engine.vision import PeerVisionTower, make_vision

class PeerVisionTest(unittest.TestCase):
    def test_head_loads_metadata_without_weights(self):
        ep=SimpleNamespace(active=True,tensor_parallel=True,world=2,rank=0)
        with patch('engine.vision._vision_args',return_value=(None,{'dim':4})), patch('engine.vision.VisionTower') as load:
            v=PeerVisionTower('unused',ep,device='cpu')
        load.assert_not_called();self.assertIsNone(v.local);self.assertEqual(v.cfg['dim'],4)

    def test_splice_transfers_complete_spans_and_preserves_text(self):
        sent=[]
        img=SimpleNamespace(start=1,types=torch.tensor([0,1,2,1,3]))
        class Local:
            def splice(self,h,images):h[1:6]=torch.arange(20).reshape(5,4)
        result=[]
        for rank in (1,0):
            v=PeerVisionTower.__new__(PeerVisionTower)
            v.ep=SimpleNamespace(rank=rank,gather_objects=lambda x:[x,x])
            v.local=Local() if rank==1 else None
            h=torch.full((7,4),-7.)
            def send(t,src):
                self.assertEqual(src,1)
                if rank==1:sent.append(t.clone())
                else:t.copy_(sent[-1])
            with patch('torch.distributed.broadcast',side_effect=send):v.splice(h,[img])
            self.assertTrue(torch.equal(h[0],torch.full((4,),-7.)))
            self.assertTrue(torch.equal(h[-1],torch.full((4,),-7.)))
            result.append(h)
        self.assertTrue(torch.equal(*result))

    def test_owner_failure_precedes_tensor_collective(self):
        v=PeerVisionTower.__new__(PeerVisionTower);v.local=None
        v.ep=SimpleNamespace(gather_objects=lambda x:[None,'owner allocation failure'])
        with patch('torch.distributed.broadcast') as send:
            with self.assertRaisesRegex(RuntimeError,'owner allocation failure'):
                v.splice(torch.zeros(3,4),[SimpleNamespace(start=0,types=torch.ones(3))])
            send.assert_not_called()

    def test_text_path_has_no_collective(self):
        v=PeerVisionTower.__new__(PeerVisionTower)
        v.ep=SimpleNamespace(gather_objects=lambda _:self.fail('text collective'))
        h=torch.ones(3,4)
        with patch('torch.distributed.broadcast') as send:self.assertIs(v.splice(h,[]),h)
        send.assert_not_called()

    def test_mode_disagreement_fails_before_loading(self):
        ep=SimpleNamespace(gather_objects=lambda _:[(True,'peer'),(True,'replicated')])
        with patch('engine.vision.VisionTower') as load:
            with self.assertRaisesRegex(ValueError,'differs'):make_vision('unused',ep)
            load.assert_not_called()

    def test_peer_load_failure_raises_on_both_ranks(self):
        ep=SimpleNamespace(gather_objects=lambda x:[x,x] if isinstance(x,tuple) else [None,'missing weights'])
        with patch('engine.vision.PeerVisionTower',return_value=object()):
            with self.assertRaisesRegex(RuntimeError,'missing weights'):
                make_vision('unused',ep,mode='peer')

if __name__=='__main__':unittest.main()
