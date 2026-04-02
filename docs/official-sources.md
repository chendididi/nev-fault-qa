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
- `python scripts/download_nhtsa_data.py --year-range 2025-2026 --make TESLA --make BYD`
  - 下载 NHTSA Manufacturer Communications（CSV/TSV）
  - 默认生成 `data/raw/official/nhtsa/chunks/nhtsa_mfr_comms_chunks.json`，可被 `build_index.py` 直接摄取
- `python scripts/fetch_tesla_service_manual.py --manual model3_2024_en_us`
  - 从本地 seed HTML 提取 Tesla service manual 链接，并抓取同站 HTML 页面到本地目录
- `python scripts/build_index.py --input data/raw/official --skip-embedding`
  - 递归摄取官方目录中的 PDF、Tesla HTML 和预构建 chunks JSON
  - 运行日志和 run manifest 会记录发现统计、跳过的非 chunks JSON，以及最终 chunk 数

## 数据源边界

- `Tesla`
  - 公开可访问的 service manual 主要是 HTML 站点
  - 可直接下载的 PDF 主要是 Service Mode Guide 和 Owner's Manual
  - 当前仓库已经支持 `HTML -> chunks`，可以直接把抓下来的 HTML 页面送进 `build_index.py`
- `BYD`
  - 当前公开可拿到的主要是 Owner's Manual / 安全资料
  - 这些资料适合作为维护、告警、故障现象补充语料，不应伪装成维修手册
- `NHTSA`
  - 官方公开的 Manufacturer Communications / TSB 数据可作为维修通告类语料补充
  - 项目当前通过 NHTSA 官方静态源下载，不依赖第三方镜像
