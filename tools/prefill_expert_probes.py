"""Deterministic, objectively graded probes for prefill arithmetic comparisons.

This is a bounded regression screen, not a general model-quality benchmark. Tool
calls are generated but never executed. Code runs with restricted syntax/builtins
in a separate, resource-limited process; syntax alone does not earn a pass.
"""
import ast
import json
import re
import subprocess
import sys


def unfence(text):
    text = text.strip()
    m = re.fullmatch(r'```(?:python|json)?\s*\n(.*?)\n```', text, re.S)
    return m.group(1).strip() if m else text


def probe(name, family, prompt, expected=None, limit=192, **extra):
    return dict(name=name, family=family, body={'messages': [{'role': 'user', 'content': prompt}]},
                expected=expected, limit=limit, **extra)


def short_probes():
    rows = []
    for depth in (4, 6, 8, 10):
        for leaf in (48, 73):
            expected = leaf
            for _ in range(depth):
                expected = {'n': expected}
            rows.append(probe(f'nest-{depth}-{leaf}', 'nesting',
                f'Output one JSON object nested exactly {depth} levels deep and nothing else. '
                'Each level has exactly one key "n" whose value is the next level down. '
                f'The innermost "n" is the integer {leaf}. '
                f'So depth 2 would be: {{"n": {{"n": {leaf}}}}}', expected))
    for i, (a, b, c) in enumerate(((17, 23, 19), (48, 7, 53), (125, 16, 87), (39, 41, 208))):
        rows.append(probe(f'arithmetic-{i}', 'arithmetic',
                          f'Calculate ({a} * {b}) - {c}. Return only a JSON object with key "answer" '
                          'and its integer value, no working.', {'answer': a*b-c}, 64))
    for i in range(4):
        names = ['Mira', 'Theo', 'Inez', 'Omar']
        n = names[i]
        expected = {'name': n, 'count': 12+i, 'active': i % 2 == 0, 'tags': ['red', 'blue']}
        rows.append(probe(f'extract-{i}', 'json',
            f'Return only JSON with exactly name, count, active, tags. Name is {n}; count is {12+i}; '
            f'active is {str(i % 2 == 0).lower()}; tags in order are red, blue. Use native JSON types.', expected))
    code = [
        ('add', 'add(a, b) that returns the sum of two integers', [([2, 3], 5), ([-4, 1], -3), ([0, 0], 0)]),
        ('clamp', 'clamp(x, lo, hi) that bounds x inclusively between lo and hi',
         [([7, 0, 5], 5), ([-2, 0, 5], 0), ([3, 0, 5], 3)]),
        ('sum_squares', 'sum_squares(xs) that returns the sum of the squares of the numbers in xs',
         [([[1, -2, 3]], 14), ([[]], 0), ([[5]], 25)]),
        ('count_even', 'count_even(xs) that counts the even integers in xs',
         [([[0, 1, -2, 3, 4]], 3), ([[]], 0), ([[1, 3, 5]], 0)]),
        ('unique_sorted', 'unique_sorted(xs) that returns the distinct integers from xs in ascending order',
         [([[3, 1, 3, -2]], [-2, 1, 3]), ([[]], []), ([[2, 2]], [2])]),
        ('is_palindrome', 'is_palindrome(s) that returns whether the string s equals its reverse, case-sensitive',
         [(['abba'], True), (['abc'], False), ([''], True)]),
    ]
    for name, task, cases in code:
        rows.append(probe('code-'+name, 'code',
            'Write only Python source defining '+task+'. No imports, annotations, examples, or explanation.',
            limit=192, function=name, cases=cases))
    for i, (name, params) in enumerate([
        ('get_weather', {'city': 'Oslo', 'unit': 'celsius'}),
        ('get_weather', {'city': 'Lima', 'unit': 'fahrenheit'}),
        ('lookup_order', {'order_id': 'A-731', 'include_items': True}),
        ('lookup_order', {'order_id': 'B-482', 'include_items': False}),
        ('convert_units', {'value': 12, 'from_unit': 'km', 'to_unit': 'm'}),
        ('convert_units', {'value': 5, 'from_unit': 'hours', 'to_unit': 'minutes'}),
    ]):
        row = probe(f'tool-{i}', 'tool',
            f'Call {name} exactly once with these arguments: {json.dumps(params)}. Do not answer in prose.',
            {'name': name, 'arguments': params})
        properties = {k: {'type': 'boolean' if isinstance(v, bool) else 'number' if isinstance(v, (int, float)) else 'string'}
                      for k, v in params.items()}
        row['body']['tools'] = [{'type': 'function', 'function': {'name': name,
            'description': 'Return the requested information.', 'parameters': {'type': 'object',
            'properties': properties, 'required': list(params), 'additionalProperties': False}}}]
        rows.append(row)
    return rows


def retrieval_probe(target_tokens, position, seed, render):
    """Fit synthetic records to context length using the real chat tokenizer.

    A unique record goes at the requested fraction, with no answer in the query.
    Each prompt has fresh records; neither arm gets prefix reuse.
    """
    target_key = f'SPECIAL-{seed}'
    secret = f'cobalt-{seed * 7919 % 1000000:06d}'
    def make(n):
        records = [f'Record {seed}-{j:05d}: project code Q{(j*73+seed)%997:03d}; '
                   f'access phrase amber-{(j*104729+seed)%1000000:06d}; status archived.\n' for j in range(n)]
        records.insert(int(n*position), f'Record {target_key}: access phrase {secret}; status current.\n')
        return probe(f'retrieval-{target_tokens}-{position}-{seed}', 'retrieval',
            'Read the records.\n'+''.join(records)+f'\nWhat is the access phrase for record {target_key}? '
            'Return only JSON with one key "phrase" and the exact phrase as its string value.',
            {'phrase': secret}, 64)
    n, best = max(8, target_tokens//30), None
    for _ in range(6):
        row = make(n)
        ids = render(row['body'])
        distance = abs(len(ids)-target_tokens)
        if best is None or distance < best[0]:
            best = distance, row, ids
        n = max(8, int(n*target_tokens/len(ids)))
    return best[1], best[2]


_CODE_RUNNER = r'''
import json, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
resource.setrlimit(resource.RLIMIT_AS, (512*1024**2, 512*1024**2))
d=json.load(sys.stdin)
b={k: __builtins__.__dict__[k] for k in ('sum','min','max','abs','len','range','enumerate','sorted','set','list','tuple','reversed','int','bool','str','zip','all','any')}
ns={'__builtins__':b}
exec(compile(d['code'], '<generated>', 'exec'), ns)
print(json.dumps([ns[d['function']](*args)==expected for args,expected in d['cases']]))
'''


def grade(row, text, enc=None):
    try:
        if row['family'] == 'tool':
            parsed = enc.parse_message_from_completion_text(text + ('' if text.endswith(enc.eos_token) else enc.eos_token),
                                                            thinking_mode='chat')
            calls = parsed.get('tool_calls') or []
            actual = []
            for call in calls:
                fn = call.get('function', call)
                args = fn['arguments']
                actual.append({'name': fn['name'], 'arguments': json.loads(args) if isinstance(args, str) else args})
            return dict(pass_=actual == [row['expected']], actual=actual)
        source = unfence(text)
        if row['family'] != 'code':
            actual = json.loads(source)
            # bool and integer compare equal in Python; JSON types must match too.
            ok = json.dumps(actual, sort_keys=True) == json.dumps(row['expected'], sort_keys=True)
            return dict(pass_=ok, actual=actual)
        tree = ast.parse(source)
        blocked = (ast.Import, ast.ImportFrom, ast.Attribute, ast.ClassDef, ast.Global, ast.Nonlocal,
                   ast.With, ast.AsyncWith, ast.AsyncFunctionDef)
        if any(isinstance(n, blocked) or isinstance(n, ast.Name) and n.id.startswith('__') for n in ast.walk(tree)):
            return dict(pass_=False, reason='unsupported code syntax')
        result = subprocess.run([sys.executable, '-I', '-c', _CODE_RUNNER],
            input=json.dumps(dict(code=source, function=row['function'], cases=row['cases'])),
            text=True, capture_output=True, timeout=4)
        if result.returncode:
            return dict(pass_=False, reason=result.stderr[-800:])
        checks = json.loads(result.stdout)
        return dict(pass_=len(checks) == len(row['cases']) and all(checks), checks=checks)
    except Exception as exc:
        return dict(pass_=False, reason=type(exc).__name__+': '+str(exc)[:400])
