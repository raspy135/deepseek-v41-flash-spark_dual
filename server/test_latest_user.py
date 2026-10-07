"""User identity and input boundaries, independent of tokenizer/model loading."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

from server.latest_user import extract_latest_user_prompt, latest_user_ranges


class LatestUserTest(unittest.TestCase):
    def test_last_actual_user_excludes_later_tool_system_and_assistant(self):
        messages = [{'role': 'system', 'content': 'policy'},
                    {'role': 'user', 'content': 'old question'},
                    {'role': 'assistant', 'content': 'old answer'},
                    {'role': 'user', 'content': 'new question'},
                    {'role': 'tool', 'content': 'new question plus retrieved data'},
                    {'role': 'system', 'content': 'late instructions'},
                    {'role': 'assistant', 'content': 'answer'}]
        got = extract_latest_user_prompt(messages)
        self.assertEqual(got.message_index, 3)
        self.assertEqual(got.text, 'new question')

    def test_structured_text_excludes_attachments_and_nested_tool_text(self):
        content = [{'type': 'text', 'text': '日本語もいける？'},
                   {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}},
                   {'type': 'file', 'file': {'filename': 'document', 'file_data': 'secret'}},
                   {'type': 'input_audio', 'input_audio': {'data': 'secret'}},
                   {'type': 'tool_result', 'content': [{'type': 'text', 'text': 'secret'}]},
                   {'type': 'text', 'text': '  preserve spacing\n'}]
        before = copy.deepcopy(content)
        got = extract_latest_user_prompt([{'role': 'user', 'content': content}])
        self.assertEqual(got.text_parts, ('日本語もいける？', '  preserve spacing\n'))
        self.assertEqual(content, before)

    def test_native_blocks_precede_legacy_content_and_exclude_tool_result(self):
        msg = {'role': 'user', 'content': 'unused fallback', 'content_blocks': [
            {'type': 'tool_result', 'content': 'retrieved document'},
            {'type': 'text', 'text': 'actual question'}]}
        self.assertEqual(extract_latest_user_prompt([msg]).text, 'actual question')

    def test_empty_or_attachment_only_latest_user_never_selects_previous(self):
        for content in ('', None, [], [{'type': 'file', 'file': {'filename': 'only file'}}]):
            with self.subTest(content=content):
                got = extract_latest_user_prompt([{'role': 'user', 'content': 'old'},
                                                  {'role': 'user', 'content': content}])
                self.assertEqual(got.message_index, 1)
                self.assertEqual(got.text, '')

    def test_no_user_does_not_treat_other_roles_as_user(self):
        self.assertIsNone(extract_latest_user_prompt([]))
        for role in ('system', 'developer', 'assistant', 'tool', 'latest_reminder', 'direct_search_results'):
            self.assertIsNone(extract_latest_user_prompt([{'role': role, 'content': 'data'}]))

    def test_plain_text_bundling_needs_client_boundary(self):
        text = 'Retrieved document: data\nActual question: summarize it'
        self.assertEqual(extract_latest_user_prompt([{'role': 'user', 'content': text}]).text, text)

    def test_offsets_identify_duplicate_unicode_text_and_exclude_tools(self):
        messages = [{'role':'user','content':'日本語'}, {'role':'assistant','content':'日本語'},
                    {'role':'user','content':'日本語'}, {'role':'tool','content':'日本語'}]
        render = lambda ms: '|'.join(m['content'] for m in ms)
        prompt = render(messages)
        offsets = [(i,i+1) for i in range(len(prompt))]
        self.assertEqual(latest_user_ranges(messages,render,prompt,offsets), [[8,11]])
        self.assertEqual(messages[2]['content'],'日本語')

    def test_native_blocks_keep_separate_ranges_and_fail_closed(self):
        messages=[{'role':'user','content_blocks':[
            {'type':'text','text':'ask'}, {'type':'tool_result','text':'data'},
            {'type':'text','text':'next'}]}]
        render=lambda ms: '|'.join(b['text'] for b in ms[0]['content_blocks'])
        prompt=render(messages)
        self.assertEqual(latest_user_ranges(messages,render,prompt,[(i,i+1) for i in range(len(prompt))]),
                         [[0,3],[9,13]])
        with self.assertRaises(ValueError):
            latest_user_ranges(messages, lambda ms:render(ms).replace('ask','ASK'), prompt, [])

    def test_tokens_mixed_with_template_are_excluded(self):
        messages=[{'role':'user','content':'ask'}]
        render=lambda ms:'<U>'+ms[0]['content']+'</U>'
        self.assertEqual(latest_user_ranges(messages,render,render(messages),[(2,4),(4,5),(5,7)]), [[1,2]])


class RealEncodingRangesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from server.app import Tok, load_encoding_module
        root = Path.home() / 'models/DeepSeek-V4.1-Flash'
        if not (root / 'tokenizer.json').exists():
            raise unittest.SkipTest('local tokenizer unavailable')
        cls.tok, cls.enc = Tok(str(root)), load_encoding_module(str(root))

    def test_real_template_unicode_duplicates_tools_and_developer(self):
        from server.app import build_chat_prompt
        messages = [{'role':'developer','content':'日本語もいける？'},
                    {'role':'user','content':'日本語もいける？'},
                    {'role':'assistant','content':'yes'},
                    {'role':'user','content':'日本語もいける？'}]
        for thinking in (False, True):
            focus = {}
            prompt, ids, _, _ = build_chat_prompt({'messages':messages}, self.enc, self.tok,
                thinking,75,SimpleNamespace(user_prompt_stream=True),focus)
            decoded = ''.join(self.tok.decode(ids[a:b]) for a,b in focus['ranges'])
            self.assertEqual(decoded,'日本語もいける？')
            self.assertEqual(len(focus['ranges']),1)
            self.assertEqual(prompt.count('日本語もいける？'),3)

    def test_real_native_tool_result_is_not_selected(self):
        from server.app import build_chat_prompt
        messages=[{'role':'user','content_blocks':[
            {'type':'tool_result','tool_name':'search','content':'exclude retrieved document'},
            {'type':'text','text':'actual question'}]}]
        focus={}
        _,ids,_,_=build_chat_prompt({'messages':messages},self.enc,self.tok,False,75,
            SimpleNamespace(user_prompt_stream=True),focus)
        self.assertEqual(''.join(self.tok.decode(ids[a:b]) for a,b in focus['ranges']), 'actual question')


if __name__ == '__main__':
    unittest.main()
