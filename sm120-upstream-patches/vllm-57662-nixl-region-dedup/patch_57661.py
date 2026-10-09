# -*- coding: utf-8 -*-
"""Apply the #57661-style fix: dedup NIXL transfer regions by (base_addr, block_len).
Upstream dev382 still folds hybrid-attention regions by base_addr alone, which
truncates DeepSeek-V4.1's compressor state (32768) with the swa_cache view (19008)."""
import re, shutil, pathlib

F = pathlib.Path('/root/vllm/lib/python3.12/site-packages/vllm/distributed/kv_transfer/'
                 'kv_connector/v1/nixl/base_worker.py')
src = F.read_text()
if '(base_addr, block_len) in seen_base_addresses' in src:
    print('already patched')
    raise SystemExit

bak = F.with_suffix('.py.bak57661')
shutil.copy2(F, bak)

orig = src
src = src.replace(
    '        seen_base_addresses: list[int] = []',
    '        seen_base_addresses: list[tuple[int, int]] = []  # (base_addr, block_len) per #57661')
src = src.replace(
    '                if base_addr in seen_base_addresses and not route_packed_layers:\n'
    '                    region_index = seen_base_addresses.index(base_addr)',
    '                region_key = (base_addr, block_len)\n'
    '                if region_key in seen_base_addresses and not route_packed_layers:\n'
    '                    region_index = seen_base_addresses.index(region_key)')
src = src.replace(
    '                else:\n'
    '                    region_index = len(seen_base_addresses)\n'
    '                    seen_base_addresses.append(base_addr)',
    '                else:\n'
    '                    region_index = len(seen_base_addresses)\n'
    '                    seen_base_addresses.append((base_addr, block_len))')

assert src != orig, 'no substitutions applied - anchor mismatch'
assert src.count('(base_addr, block_len)') >= 2
F.write_text(src)
print('patched', F)
print('backup', bak)
# syntax check
import py_compile
py_compile.compile(str(F), doraise=True)
print('syntax OK')
