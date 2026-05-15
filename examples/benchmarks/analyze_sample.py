import sys, json

data = json.loads(sys.stdin.read())
prof = data.get('engine_profile', {})

for k in sorted(prof.keys()):
    if 'sample' in k.lower():
        print(f'  {k}: {prof[k]}')
