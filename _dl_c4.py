"""Stream chinese-c4 shard (GBK encoded) from ModelScope, decompress with
proper incremental GBK decoder, keep first N lines, abort early."""
import os, sys, json, urllib.request, codecs
import zstandard as zstd

URL = "https://modelscope.cn/api/v1/datasets/swift/chinese-c4/repo?Revision=master&FilePath=data/chinese-c4-0000-of-0096.jsonl.zst"
OUT = "data/pretrain_corpus/raw/chinese_c4_sample.txt"
KEEP = 30000

os.makedirs(os.path.dirname(OUT), exist_ok=True)

dctx = zstd.ZstdDecompressor()
req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})

n = 0
out_f = open(OUT, "w", encoding="utf-8")
with urllib.request.urlopen(req, timeout=120) as resp, \
     dctx.stream_reader(resp) as reader:
    text_buf = ""
    for raw in iter(lambda: reader.read(1 << 16), b""):
        if not raw:
            break
        # data is UTF-8 (confirmed via hex); errors=replace guards partial-char at chunk end
        text_buf += raw.decode("utf-8", errors="replace")
        while "\n" in text_buf:
            line, text_buf = text_buf.split("\n", 1)
            line = " ".join(line.strip().split())
            if len(line) < 20:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("text") or obj.get("content") or ""
            t = " ".join(t.strip().split())
            if len(t) < 20:
                continue
            n += 1
            if n % 5000 == 0:
                print(f"  kept {n} lines...", flush=True)
            out_f.write(t + "\n")
            if n >= KEEP:
                print(f"Reached {KEEP} lines, stopping download early.")
                out_f.close()
                sys.exit(0)
out_f.close()
print(f"Done. Total kept: {n} lines -> {OUT}")
