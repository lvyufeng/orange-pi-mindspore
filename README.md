# MindSpore models on Orange Pi AI Pro

## Dependency

1. CANN 7.0
2. CANN Kernel 7.0

## How to run

### Install MindSpore

1. install from pypi
```bash
# not support now
pip install mindspore==2.2.12
```

2. install from daily whl

Found whl package from [url](https://repo.mindspore.cn/mindspore/mindspore/daily/202403/20240305/r2.2_20240305041520_23a0be836d6a7f3bf7d04a3372e4e7539a265899/unified/aarch64/), and download.

```bash
pip install mindspore-2.2.12*.whl
```
### Set environment variables

```bash
source env.sh
```

### RUN model

For mindspore official examples:

```bash
python mindspore_*.py
```

For LLM:

```
cd uie
python uie_predictor.py -m uie-base --use_fp16 -g
```

If your available memory is not enought, use this command:

```
bash free_mem.sh
```
