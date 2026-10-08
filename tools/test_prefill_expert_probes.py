import json
import unittest
from tools.prefill_expert_probes import grade, short_probes


class Graders(unittest.TestCase):
    def test_nesting_and_json_require_exact_structure_and_types(self):
        for p in short_probes():
            if p['family'] in ('nesting', 'json', 'arithmetic'):
                self.assertTrue(grade(p, json.dumps(p['expected']))['pass_'], p['name'])
                self.assertFalse(grade(p, '{}')['pass_'], p['name'])
        p=next(p for p in short_probes() if p['family']=='json')
        bad=dict(p['expected'],active=1)
        self.assertFalse(grade(p,json.dumps(bad))['pass_'])

    def test_python_is_executed_and_wrong_results_fail(self):
        p=next(p for p in short_probes() if p['name']=='code-add')
        self.assertTrue(grade(p,'def add(a, b):\n    return a+b')['pass_'])
        self.assertFalse(grade(p,'def add(a, b):\n    return a-b')['pass_'])
        self.assertFalse(grade(p,'def add(a, b) return a+b')['pass_'])
        self.assertFalse(grade(p,'import os\ndef add(a,b): return a+b')['pass_'])

    def test_tool_arguments_are_graded_without_execution(self):
        p=next(p for p in short_probes() if p['family']=='tool')
        expected=p['expected']
        class Encoding:
            eos_token='EOS'
            def parse_message_from_completion_text(self,text,thinking_mode):
                return {'tool_calls':[{'function':{'name':expected['name'],
                        'arguments':json.dumps(expected['arguments'])}}]}
        self.assertTrue(grade(p,'synthetic',Encoding())['pass_'])
        p=dict(p,expected=dict(name='wrong',arguments={}))
        self.assertFalse(grade(p,'synthetic',Encoding())['pass_'])


if __name__=='__main__':
    unittest.main()
