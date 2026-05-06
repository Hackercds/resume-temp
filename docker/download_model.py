"""Docker 构建时预下载 BGE 模型到本地目录"""
import os, sys, glob

proxy = os.environ.get('HTTP_PROXY', '')
print(f'代理: {proxy if proxy else "无"}')

local = '/build/bge-model'
print(f'目标路径: {local}')

from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-small-zh-v1.5', local_dir=local,
                  local_files_only=False, resume_download=True)

files = glob.glob(local + '/**', recursive=True)
print(f'下载文件数: {len(files)}')

for f in ['config.json', 'model.safetensors', 'pytorch_model.bin',
          'tokenizer.json', 'sentence_bert_config.json', 'modules.json']:
    fp = os.path.join(local, f)
    if os.path.exists(fp):
        print(f'  OK {f} ({os.path.getsize(fp)} bytes)')
    else:
        print(f'  MISS {f}')
