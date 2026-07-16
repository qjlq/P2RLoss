#!/usr/bin/env python3
import re
import sys

def is_warning(line):
    warnings = ['FutureWarning', 'UserWarning', 'DeprecationWarning', 'ImportWarning']
    return any(w in line for w in warnings)

def is_pure_progress_bar(line):
    if 'INFO' in line:
        return False
    stripped = line.strip().strip('\r')
    if not stripped:
        return True
    patterns = [
        r'^Epoch \[\d+\]:\s*\d+%.*\|.*\[.*\]',
        r'^Overall:\s*\d+%.*\|.*\[.*\]',
        r'^Test:\s*\d+%.*\|.*\[.*\]',
        r'^\s*$',
    ]
    for p in patterns:
        if re.match(p, stripped):
            return True
    return False

def extract_info(line):
    info_match = re.search(r'(\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \w+\]\([^)]+\): INFO.*)', line)
    if info_match:
        return info_match.group(1)
    return None

def clean_ansi(text):
    text = re.sub(r'\r', '', text)
    text = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
    return text

def process_file(input_path, output_path):
    total = 0
    kept = 0
    removed = 0
    with open(input_path, 'r', encoding='utf-8', errors='replace') as f_in:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            prev_empty = False
            for raw_line in f_in:
                total += 1
                line = clean_ansi(raw_line).rstrip()
                if is_warning(raw_line):
                    removed += 1
                    continue
                if is_pure_progress_bar(raw_line):
                    removed += 1
                    continue
                extracted = extract_info(line)
                if extracted:
                    if not prev_empty:
                        f_out.write('\n')
                    f_out.write(extracted + '\n')
                    prev_empty = False
                    kept += 1
                    continue
                if not line:
                    if prev_empty:
                        continue
                    prev_empty = True
                    kept += 1
                    f_out.write('\n')
                else:
                    prev_empty = False
                    kept += 1
                    f_out.write(line + '\n')
    return total, kept, removed

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input> <output>")
        sys.exit(1)
    total, kept, removed = process_file(sys.argv[1], sys.argv[2])
    print(f"Done: {total} lines → {kept} kept, {removed} removed ({kept/total*100:.1f}% kept)")
