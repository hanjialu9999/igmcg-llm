r"""
fetch_ms_corpus.py — 从魔搭(ModelScope)下载多领域预训练语料并合并进 merged.txt。

用法:
    F:\Projects\.my_venv\Scripts\python.exe scripts/data/fetch_ms_corpus.py

策略(2026-07-14):
    - 多领域中文语料，补充到约 600MB 总训练数据:
        * gongjy/minimind_dataset 各域 jsonl: agent_rl / agent_rl_math /
          dpo / lora_exam / lora_medical / rlaif (工具调用/数学/偏好/教育/医疗/RLHF)
        * AI-ModelScope/wikipedia-cn-20230720-filtered 维基百科中文(百科域, 截断到 wiki_cap_mb)
    - 下载到 data/pretrain_corpus/raw/
    - 调用 download_pretrain_data.py --prepare 转成训练 txt
    - 把“本次新增的 txt”追加到 merged.txt (补充而非重建，避免与既有数据重复)
"""
import os, sys, time, glob, json

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(ROOT, "data", "pretrain_corpus", "raw")
OUT_DIR = os.path.join(ROOT, "data", "pretrain_corpus")

WIKI_CAP_MB = 400  # 维基百科截断大小(MB)

SMALL_FILES = [
    ("gongjy/minimind_dataset", "agent_rl_math.jsonl"),
    ("gongjy/minimind_dataset", "dpo.jsonl"),
    ("gongjy/minimind_dataset", "lora_exam.jsonl"),
    ("gongjy/minimind_dataset", "rlaif.jsonl"),
]
WIKI = ("AI-ModelScope/wikipedia-cn-20230720-filtered",
        "wikipedia-cn-20230720-filtered.jsonl")


def download_small(dataset, fname):
    from modelscope.hub.file_download import dataset_file_download
    out = os.path.join(RAW_DIR, fname)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"[跳过] {fname} 已存在 ({os.path.getsize(out)/1e6:.1f}MB)")
        return out
    t = time.time()
    p = dataset_file_download(dataset, fname, local_dir=RAW_DIR)
    dt = time.time() - t
    print(f"[完成] {fname} {os.path.getsize(p)/1e6:.1f}MB 用时 {dt:.1f}s "
          f"速度 {os.path.getsize(p)/1e6/dt:.2f}MB/s")
    return p


def download_capped(dataset, fname, cap_mb):
    import requests
    from modelscope.hub.api import HubApi
    out = os.path.join(RAW_DIR, fname)
    cap = int(cap_mb * 1e6)
    if os.path.exists(out) and os.path.getsize(out) >= cap:
        print(f"[跳过] {fname} 已存在 ({os.path.getsize(out)/1e6:.1f}MB >= {cap_mb}MB)")
        return out
    namespace, dataset_name = dataset.split("/", 1)
    url = HubApi().get_dataset_file_url(fname, dataset_name, namespace)
    print(f"[下载] {fname} 截断到 {cap_mb}MB ...")
    t = time.time()
    written = 0
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                if written + len(chunk) >= cap:
                    rem = cap - written
                    if rem > 0:
                        f.write(chunk[:rem])
                        written += rem
                    break
                f.write(chunk)
                written += len(chunk)
    # 截到最后一个换行符，避免尾部残留半个多字节字符导致解码失败
    with open(out, "rb") as f:
        data = f.read()
    idx = data.rfind(b"\n")
    if idx > 0:
        data = data[: idx + 1]
    with open(out, "wb") as f:
        f.write(data)
    written = len(data)
    dt = time.time() - t
    print(f"[完成] {fname} {written/1e6:.1f}MB 用时 {dt:.1f}s "
          f"速度 {written/1e6/dt:.2f}MB/s")
    return out


def snapshot_txt():
    s = set()
    for p in glob.glob(os.path.join(OUT_DIR, "*.txt")):
        s.add(os.path.abspath(p))
    return s


# 本次新增、且“原本不在 merged.txt 中”的语料文件(相对路径名)
NEW_FILES = [
    "agent_rl.txt",
    "agent_rl_math.txt",
    "lora_exam.txt",
    "lora_medical.txt",
    "rlaif.txt",
    "wikipedia-cn-20230720-filtered.txt",
]
# 原始 merged.txt 的字节数(本次操作前的大小，作为“基底”保留，避免重复)
ORIGINAL_MERGED_BYTES = 100707122


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    print("== 1) 下载 ==")
    for ds, fn in SMALL_FILES:
        download_small(ds, fn)
    download_capped(*WIKI, WIKI_CAP_MB)

    print("== 2) 转换 jsonl->txt ==")
    sys.path.insert(0, os.path.join(ROOT, "scripts", "data"))
    import download_pretrain_data as dpd
    dpd.prepare_raw_data()

    print("== 3) 重建 merged.txt ==")
    merged = os.path.join(OUT_DIR, "merged.txt")
    # 基底 = 原始 merged.txt 的前 ORIGINAL_MERGED_BYTES 字节(未被本次追加污染)
    with open(merged, "rb") as f:
        base = f.read(ORIGINAL_MERGED_BYTES)
    print(f"[基底] 原始 merged.txt {len(base)/1e6:.1f}MB 保持不变")

    # 仅追加本次新增且原本不在基底中的文件(排除 chinese_c4_sample/unknow_zh 等已含于基底者)
    total_new = 0
    with open(merged, "wb") as fout:
        fout.write(base)
        for name in NEW_FILES:
            p = os.path.join(OUT_DIR, name)
            if not os.path.exists(p):
                print(f"[警告] 缺少 {name}，跳过")
                continue
            sz = os.path.getsize(p)
            with open(p, "rb") as fin:
                while True:
                    buf = fin.read(1 << 20)
                    if not buf:
                        break
                    fout.write(buf)
            total_new += sz
            print(f"  + {name} {sz/1e6:.1f}MB")
    new_size = os.path.getsize(merged)
    print(f"[merged.txt] 基底 {len(base)/1e6:.1f}MB + 新增 {total_new/1e6:.1f}MB "
          f"= {new_size/1e6:.1f}MB")


if __name__ == "__main__":
    main()
