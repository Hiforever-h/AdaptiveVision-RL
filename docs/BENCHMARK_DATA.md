# Benchmark image pairs

The requested evaluation data lives in seven separate directories under `data/`.
Each question in `samples.jsonl` points to one original image and one image
whose width and height are each `floor(original / 2)` (with a minimum of one
pixel). The latter therefore has approximately one quarter as many pixels.
When questions share an image, their records point to the same image files.

| Directory | Public split | Questions | Source |
| --- | --- | ---: | --- |
| `data/ChartQA` | test | 2,500 | [HuggingFaceM4/ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) |
| `data/OCRBench` | test | 1,000 | [echo840/OCRBench](https://huggingface.co/datasets/echo840/OCRBench) |
| `data/MME` | test | 2,374 | [lmms-lab/MME](https://huggingface.co/datasets/lmms-lab/MME) |
| `data/MMVet` | test | 218 | [whyu/mm-vet](https://huggingface.co/datasets/whyu/mm-vet) |
| `data/RealWorldQA` | test | 765 | [xai-org/RealworldQA](https://huggingface.co/datasets/xai-org/RealworldQA) |
| `data/POPE` | test, all three categories | 9,000 | [lmms-lab/POPE](https://huggingface.co/datasets/lmms-lab/POPE) |
| `data/MathVerse` | testmini, all five visual versions | 3,940 | [AI4Math/MathVerse](https://huggingface.co/datasets/AI4Math/MathVerse) |

MathVerse only publishes `testmini` with images and answers. The DocVQA test
split and MathVista full test split omit public answers, so they were excluded
as requested. Their downloaded files and derived images were removed.
In MathVerse's Vision Only variant, the source intentionally leaves its
`question` field empty because the full problem is in the image. The manifest's
top-level `question` uses the source's `query_wo` instruction for those rows;
the source field remains unchanged in `data`.

Within each directory:

- `source.json` records the source URL, pinned revision, split, and file sizes.
- `source/` contains the downloaded source Parquet files.
- `images/high/` contains the embedded image bytes without resizing or re-encoding.
- `images/low/` contains the resized images, using Lanczos interpolation.
- `samples.jsonl` contains one record per question, including `question`,
  `answer`, `high_image`, `low_image`, both image sizes, the original image hash,
  and the source row's remaining fields in `data`.

Image paths in the manifest are relative to that dataset's own directory.
For example, `data/ChartQA/samples.jsonl` entries resolve against
`data/ChartQA/`. Generated low-resolution images apply EXIF orientation before
resizing; original image bytes remain unchanged. JPEG and WebP low-resolution
images are re-encoded at quality 95; PNG images are saved as PNG.

To reproduce or check the preparation, use the project's `vison-rl` Python
environment from the repository root:

```bash
python scripts/prepare_benchmarks.py download
python scripts/prepare_benchmarks.py extract
python scripts/prepare_benchmarks.py verify
```

The `verify` command checks row counts, nonempty questions and answers, the
presence of both images, and every image pair's dimensions.

## Frozen project evaluation subset

`data/eval_6bench_3000_v1/` is a self-contained subset for evaluating training
results. Its six subdirectories each contain a `samples.jsonl`, copied
high/low image pairs, and `upstream_source.json`. Paths in each manifest are
relative to its own dataset subdirectory. `selection.json` records counts,
strata, the fixed seed `20260929`, source manifest checksums, and the image
overlap audit. The subset is frozen; the sampling script refuses to overwrite
an existing output.

| Dataset | Selected questions | Selection rule |
| --- | ---: | --- |
| ChartQA | 500 | 250 human and 250 machine questions; distinct images |
| OCRBench | 500 | 50 from each of ten question types; distinct images |
| MME | 500 | 250 distinct-image Yes/No pairs, spread across 14 categories |
| RealWorldQA | 500 | Random questions with distinct images |
| POPE | 500 | 250 distinct-image Yes/No pairs across three categories |
| MathVerse | 500 | 100 from each visual version, distinct base problems and images |

The six manifests total 3,000 questions. MMVet is excluded because its
free-response grading requires an external judge. Selection checked image byte and
pixel hashes against the existing VisionThink and pilot manifests. One POPE
candidate image matched those data at the pixel level and was replaced before
freezing; the final subset has no such matches. This audit does not establish
absence of overlap with a model's unknown pretraining data.

Reproduce or verify the frozen selection with:

```bash
python scripts/sample_benchmark_eval.py create
python scripts/sample_benchmark_eval.py verify
```

`create` is for an empty output path; `verify` checks the current frozen copy.
This step prepares evaluation inputs only. Model inference and answer scoring
remain separate.
