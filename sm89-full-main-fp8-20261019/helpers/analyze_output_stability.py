"""Compare same-arm repeatability with cross-arm output differences."""
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'results' / 'tp2'
arms = ['base-a1', 'candidate-b1', 'candidate-b2', 'base-a2']
data = {a: json.loads((root / 'http' / f'{a}.json').read_text()) for a in arms}
tokens = {a: {(r['concurrency'], r['repeat'], i['index']): i['output_token_ids']
              for r in data[a]['rounds'] for i in r['records']} for a in arms}
result = {'pairs': [], 'unique_outputs': {}, 'quality_variations': []}
for left, right in combinations(arms, 2):
    for c in [1, 2, 8]:
        keys = [k for k in tokens[left] if k[0] == c]
        first = Counter(next((i for i, (x, y) in enumerate(zip(tokens[left][k], tokens[right][k]))
                              if x != y), 128) for k in keys)
        result['pairs'].append({'pair': [left, right], 'concurrency': c,
            'requests': len(keys), 'exact': first[128], 'first_difference_token_zero_based': dict(first)})
for a in arms:
    result['unique_outputs'][a] = {str(c): len({tuple(v) for k, v in tokens[a].items() if k[0] == c})
                                    for c in [1, 2, 8]}
quality = {a: json.loads((root / f'{a}-quality.json').read_text())['records'] for a in arms}
for i in range(16):
    texts = {a: quality[a][i]['response']['choices'][0]['message']['content'] for a in arms}
    if len(set(texts.values())) > 1:
        result['quality_variations'].append({'index': i, 'texts': texts})
(root / 'output-stability.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
