# Official Sources

这个项目默认只接入公开可访问、来源清晰的官方资料。

## 原则

- 下载只落到 `data/raw/official/`
- 原始文档和大规模切分结果只用于本地内部知识库构建
- 默认不使用来源不清的 GitHub / HuggingFace 维修手册镜像

## 当前自动化入口

- `python scripts/download_official_docs.py`
  - 默认下载官方 PDF
  - Tesla 官方 service manual 的 HTML 入口会保存在 manifest 中，按需可用 `--include-html` 抓取入口页快照

## 数据源边界

- `Tesla`
  - 公开可访问的 service manual 主要是 HTML 站点
  - 可直接下载的 PDF 主要是 Service Mode Guide 和 Owner's Manual
- `BYD`
  - 当前公开可拿到的主要是 Owner's Manual / 安全资料
  - 这些资料适合作为维护、告警、故障现象补充语料，不应伪装成维修手册
