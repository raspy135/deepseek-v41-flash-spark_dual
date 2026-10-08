import unittest
from engine.prefill_budget import PrefillBudget

class BudgetTests(unittest.TestCase):
    def test_tighter_rank_and_floor(self):
        p=PrefillBudget(True,4,128,2)
        r=p.choose(1024,55000,[8*2**30,5*2**30])
        self.assertEqual(r['rows'],256)
        self.assertTrue(r['estimate_fits'])
        self.assertEqual(r,p.choose(1024,55000,[5*2**30,8*2**30]))
    def test_recovery_and_long_context(self):
        p=PrefillBudget(True)
        self.assertEqual(p.choose(1024,1000,[10*2**30])['rows'],1024)
        self.assertLess(p.choose(1024,200000,[6*2**30])['rows'],1024)
    def test_floor_is_not_a_promise(self):
        p=PrefillBudget(True)
        for avail in [None,0,3*2**30]:
            r=p.choose(1024,55000,[avail])
            self.assertEqual(r['rows'],256)
            self.assertFalse(r['estimate_fits'])
    def test_disabled_and_small_maximum(self):
        self.assertEqual(PrefillBudget().choose(512,55000,[None])['rows'],512)
        self.assertEqual(PrefillBudget(True).choose(64,55000,[0])['rows'],64)

if __name__=='__main__':unittest.main()
