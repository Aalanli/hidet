import atexit

# clean the cache dir in envorioment variable TRITON_CACHE_DIR
import os
import shutil

if 'TRITON_CACHE_DIR' not in os.environ:
    os.environ['TRITON_CACHE_DIR'] = './triton_cache'
cache_dir = os.environ['TRITON_CACHE_DIR']
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

# dump ir
os.environ['MLIR_ENABLE_DUMP'] = '1'

# redict stderr to a file
import sys

fname = 'triton_stderr.log'
new_stderr = open(fname, 'w')
os.dup2(new_stderr.fileno(), sys.stderr.fileno())


def process_content(content):
    import re
    # remove (#loc\digit*)
    content = re.sub(r'loc\(#loc\d+\)', '', content)

    # remove loc(.*)
    content = re.sub(r'loc\([^:]*:\d+:\d+\)', '', content)

    # remove #loc.*
    content = re.sub(r'#loc.*', '', content)

    # remove empty lines
    content = '\n'.join([line for line in content.split('\n') if len(line.strip()) > 0])

    return content


def extract_file_name(head_line):
    import re
    print(head_line)
    a, b = re.search(
        # r'IR Dump (Before|After) \w+ \((\w+)\) \(.* @(\w+)\)',
        r'IR Dump (Before|After) \w+ \(([\w-]+)\)',
        head_line
    ).group(1, 2)
    print(a, b)
    return f'{a.lower()}-{b}'


def run_at_exit():
    import re
    new_stderr.close()

    lines = []
    with open(fname) as f:
        lines = list(f.readlines())
    current_lines = []
    # the pattern that starts with "//" and ends with "//"
    pattern = re.compile(r'//.*//')
    idx = 0
    shutil.rmtree('./ir', ignore_errors=True)
    os.makedirs('./ir', exist_ok=True)
    previous_content = ''
    for i, line in enumerate(lines):
        if not pattern.match(line):
            current_lines.append(line)
        else:
            if len(current_lines) > 0:
                head_line = current_lines[0]
                file_name = extract_file_name(head_line)
                # file_name = ''
                current_content = ''.join(current_lines[1:])
                current_content = process_content(current_content)
                with open(f'ir/{idx}{"*" if previous_content != current_content else ""}-{file_name}.mlir', 'w') as f:
                    f.write(head_line)
                    f.write(current_content)
                previous_content = current_content
                idx += 1
            current_lines = [line]
    with open(f'ir/{idx}.txt', 'w') as f:
        f.writelines(current_lines)


atexit.register(run_at_exit)
