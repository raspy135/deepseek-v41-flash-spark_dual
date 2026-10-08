import os,unittest
from unittest.mock import patch
from engine.spec_depth import ConfidenceDepthPolicy

class RefreshTests(unittest.TestCase):
 def test_stale_width_is_remeasured_without_proposal_lookahead(self):
  with patch.dict(os.environ,{'DSV41_CONF_COST_REFRESH':'8'}):p=ConfidenceDepthPolicy()
  p.step_s={1:.074,3:.155,5:.110}
  for _ in range(8):p.observe(1,1,2,.074)
  # Refresh choice cannot depend on the proposals/confidences that are excluded.
  self.assertEqual(p.choose_sampled([100]*5),3)
  self.assertEqual(p.choose_sampled([-100]*5),3)
  self.assertEqual(p.choose_sampled([float('nan')]*5),3)
  p.observe(3,3,4,None) # graph capture is not a cost sample
  self.assertEqual(p._refresh_samples,0)
  for _ in range(3):p.observe(3,3,4,.095)
  self.assertAlmostEqual(p.step_s[3],.095)
  self.assertIsNone(p._refresh_depth)
 def test_disabled_and_pinned(self):
  with patch.dict(os.environ,{'DSV41_CONF_COST_REFRESH':'0'}):p=ConfidenceDepthPolicy()
  p._cost_age[3]=10000
  self.assertIsNone(p._refresh_target())
  p.refresh_interval=1;p.pinned=5
  self.assertEqual(p.choose_sampled([0]*5),5)
  self.assertEqual(p.choose([0]*5),5)

if __name__=='__main__':unittest.main()
