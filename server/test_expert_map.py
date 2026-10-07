"""Observer tests: no GPU, no inference locks, no prompt content."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
import json
import threading
import unittest
import urllib.request
from types import SimpleNamespace as NS

from engine.expert_activity import ExpertActivity
from server.expert_map import snapshot


def fixture():
    store=NS(lru=OrderedDict({(0,0):0,(0,1):1,(1,2):2}),transient_map={(1,3):3},
             activity=ExpertActivity(),arena=NS(slots=6),null_slot=5,transient_slots=2)
    return NS(store=store,args=NS(n_layers=2,n_routed_experts=4),expert_bytes=9400320,
              expert_generation=2,ep=NS(rank=0,world=2,tensor_parallel=True),
              dynamic_experts=True,user_prompt_stream=False,
              last_stats={'prune_miss_request':{'miss_rate':.125},'private_prompt':'must not leak'},
              config=lambda:(_ for _ in ()).throw(AssertionError('No GPU configuration reads')))


class ExpertMapTest(unittest.TestCase):
    def test_full_directory_and_loading_override(self):
        engine=fixture()
        with engine.store.activity.loading((0,2),1):
            d=snapshot(engine,busy=True)
            self.assertEqual(d['states'],['1030','0012'])
            self.assertEqual(d['slots'][0],[0,-1,1,-1])
            self.assertEqual(d['sectors'][1],[0,2,3])
            self.assertEqual(d['counts'],dict(not_loaded=4,resident=2,transient=1,loading=1))
            self.assertTrue(d['busy'])
            self.assertEqual(d['last_request_miss_rate'],.125)
            self.assertNotIn('must not leak',json.dumps(d))
        d=snapshot(engine)
        self.assertEqual(d['active'],[])
        self.assertEqual(d['recent'][0]['expert'],2)
        self.assertTrue(d['recent'][0]['ok'])

    def test_failure_events_and_bounded_history(self):
        a=ExpertActivity(capacity=2)
        with self.assertRaises(ValueError):
            with a.loading((0,1),3):raise ValueError('failed')
        self.assertFalse(a.snapshot()['recent'][0]['ok'])
        for e in (2,3):
            with a.loading((0,e),e):pass
        d=a.snapshot()
        self.assertEqual(d['sequence'],3)
        self.assertEqual(len(d['recent']),2)
        self.assertEqual(d['active'],[])

    def test_snapshot_does_not_wait_for_weight_io(self):
        a=ExpertActivity();entered=threading.Event();release=threading.Event()
        def load():
            with a.loading((1,2),3):entered.set();release.wait(2)
        with ThreadPoolExecutor(1) as pool:
            future=pool.submit(load)
            self.assertTrue(entered.wait(1))
            try:self.assertEqual(a.snapshot()['active'][0]['expert'],2)
            finally:release.set()
            future.result()

    def test_http_page_and_json_while_generation_lock_is_held(self):
        from server.app import Handler
        lock=threading.Lock();lock.acquire()
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        server.state=NS(engine=fixture(),lock=lock,ep_fault=None,model_name='deepseek')
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base='http://127.0.0.1:'+str(server.server_port)
        try:
            with urllib.request.urlopen(base+'/expert-map',timeout=2) as r:
                self.assertEqual(r.headers.get_content_type(),'text/html')
                self.assertIn(b'By arena sector',r.read())
            with urllib.request.urlopen(base+'/v1/expert-map',timeout=2) as r:
                d=json.load(r)
                self.assertTrue(d['busy'])
                self.assertEqual(len(d['states']),2)
                self.assertEqual(d['counts']['resident'],3)
        finally:
            server.shutdown();server.server_close();thread.join();lock.release()

    def test_unsupported_engine(self):
        self.assertIsNone(snapshot(NS()))


if __name__=='__main__':unittest.main()
