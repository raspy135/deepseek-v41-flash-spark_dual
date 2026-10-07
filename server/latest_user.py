"""Select user-authored text before the encoder merges tools and rewrites roles.

This helper extracts text only. It does not alter model input, record prompt
contents, or change expert selection. Plain-text documents bundled into the
same field cannot be distinguished from the user's question without a boundary.
"""
from dataclasses import dataclass
import copy
import uuid


@dataclass(frozen=True)
class LatestUserPrompt:
    message_index: int
    text_parts: tuple[str, ...]

    @property
    def text(self):
        return '\n\n'.join(self.text_parts)


def extract_latest_user_prompt(messages):
    """Return text from the last original role=user, never fall back to older text.

    System/developer, assistant, tool, reminder and search messages are excluded.
    In structured content, only explicit text blocks are selected; attachments,
    images, audio and nested tool-result text are excluded. The empty latest
    user message produces an empty selection instead of selecting an older one.
    Native content_blocks take precedence as they do in the model encoder.
    """
    for index in range(len(messages) - 1, -1, -1):
        msg = messages[index]
        if not isinstance(msg, dict) or msg.get('role') != 'user':
            continue
        native = msg.get('content_blocks')
        content = native if isinstance(native, list) and native else msg.get('content')
        if isinstance(content, str):
            parts = (content,)
        elif isinstance(content, list):
            parts = tuple(block['text'] for block in content
                          if isinstance(block, dict)
                          and block.get('type') in ('text', 'input_text')
                          and isinstance(block.get('text'), str))
        else:
            parts = ()
        return LatestUserPrompt(index, parts)
    return None


def latest_user_ranges(messages, render, prompt, offsets):
    """Locate original user text in the *verified* rendering, then token offsets.

    Duplicate text, merged tool messages and mid-chat system messages are safe:
    only the original user fields get markers, which never reach the model.
    A rendering changed by the markers fails closed rather than guessing.
    """
    selected = extract_latest_user_prompt(messages)
    if selected is None or not any(selected.text_parts):
        return []
    marked = copy.deepcopy(messages)
    msg = marked[selected.message_index]
    native = msg.get('content_blocks')
    key = 'content_blocks' if isinstance(native, list) and native else 'content'
    nonce = uuid.uuid4().hex
    pairs = []

    def wrap(text):
        if not text:
            return text
        a, b = f'\ue000{nonce}:{len(pairs)}:a\ue001', f'\ue000{nonce}:{len(pairs)}:b\ue001'
        pairs.append((a, b))
        return a + text + b

    if isinstance(msg.get(key), str):
        msg[key] = wrap(msg[key])
    else:
        for block in msg.get(key) or []:
            if (isinstance(block, dict) and block.get('type') in ('text', 'input_text')
                    and isinstance(block.get('text'), str)):
                block['text'] = wrap(block['text'])
    rendered = render(marked)
    chars, clean, cursor = [], '', 0
    for a, b in pairs:
        if rendered.count(a) != 1 or rendered.count(b) != 1:
            raise ValueError('latest-user markers were changed by the encoder')
        left = rendered.index(a, cursor)
        right = rendered.index(b, left + len(a))
        clean += rendered[cursor:left]
        start = len(clean)
        clean += rendered[left + len(a):right]
        chars.append((start, len(clean)))
        cursor = right + len(b)
    clean += rendered[cursor:]
    if clean != prompt:
        raise ValueError('latest-user marked rendering differs from model input')
    # Boundary tokens containing template text are deliberately excluded.
    picked = [i for i, (a, b) in enumerate(offsets)
              if b > a and any(lo <= a and b <= hi for lo, hi in chars)]
    ranges = []
    for i in picked:
        if ranges and ranges[-1][1] == i:
            ranges[-1][1] = i + 1
        else:
            ranges.append([i, i + 1])
    return ranges
