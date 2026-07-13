import os
os.chdir(r'F:\Projects\igmcg-llm')
with open('scripts/generate.py', 'r', encoding='utf-8') as f:
    c = f.read()
    for i, line in enumerate(c.split(chr(10))):
        if 'def generate_igmcg' in line:
            print(i+1, line)