# EQUIVISION · Equine fetlock radiograph workbench

[简体中文](README.md) · **English**

[Padding study and per-film results](image-study/published/cv84/canvas-round2/README.md#english) · [Research guide](image-study/README.md)

The Python workbench supports team case files, image review, whole-film screening, lesion localization and signed reports. Both inference models are included in the repository and run on the server's CPU. A separate GitHub Pages site runs YOLO in the visitor's browser and stores its history in that browser. [Research workbench models](backend/models) · [Browser demo](https://elfaw775.github.io/sport-model/)

## How backend edits affect model output

The Python workbench handles visits, sign-in, report signing and manual annotations. The same server process loads two ONNX files, verifies their SHA-256 hashes and runs inference on CPU. Editing forms, case logic or saving an annotation does not write model weights. Training and export run separately. [Deployed files and DINO threshold](backend/models) · [Model loader](backend/onnx_models.py)

Changing resizing, padding, input layout, operating thresholds or box filtering can change results with identical weight files. Compare scores, screening calls and boxes on fixed reference films when editing inference. The reader can open numbered AI boxes or return to the original primary film directly from the analysis guidance. [Screening](backend/screening.py) · [Localization](backend/detector.py) · [Regression tests](backend/tests/test_detector.py)

[Box-supervised experiment](image-study/published/cv84/box-token/README.md#english) · [Reward experiment and lessons from published methods](image-study/published/cv84/one-step-reward/README.md#english)

## Run the case workbench

From the repository root, create a Python environment and start FastAPI on the loopback interface.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.txt
.venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8765
```

On Windows, use `.venv\Scripts\python.exe` for the last two commands. Open the local workbench and choose a demo member. The usernames are `wangyiyi`, `zhushijia`, `alimire` and `liqihang`. The local demo password is the username followed by `sportsHorse`. Members share team cases, while the case retains its creator and the report retains its signer. [Open the local workbench](http://127.0.0.1:8765/)

### Review a case

For a live demonstration, click **随机抽取样片** on the new examination form to draw from the 84 project radiographs. Preview the film or draw another before uploading. Check the projection and image quality, upload to obtain a case number, then run both models, review the findings, sign and archive the report. The **推荐病例** section also offers films 16 and 23 for repeatable demonstrations. These radiographs were used during model development.

1. Create a visit, enter the horse and projection, upload a PNG, JPEG or WebP and confirm image quality. A visit may also start without images. The first film is the primary view. Later views support side-by-side review. [Example lateral film](training/data/sporthorse_v2/images/test/16-侧位.jpg)
2. Analyze the primary film. DINOv2 Small returns a whole-film screening result. YOLO11s locates candidate regions. Each has its own threshold and model version. If either inference fails, the original stays available for a retry.
3. Pan, zoom, rotate or invert the image. Record two-point distances in pixels and three-point angles in degrees. On any view, draw a rectangle or polygon, name the region and save it as a manual annotation. Toggle the overlay to inspect the original. A PNG export contains numbered outlines at original resolution, with region labels and coordinates embedded as PNG text metadata.
4. Save draft findings or a conclusion, then sign the report. A deliberately empty report needs explicit confirmation. Changed drafts and annotation versions are checked before signing. Signed reports and manual annotations are read-only. Archiving is a separate action. The dashboard can reopen a case and its signed report.

The new-visit form also offers two built-in research films. Both follow the regular upload and inference path. In the review queue, filter cases by their stored screening and localization results, including positive screens with no boxes. The report checks the displayed case and image evidence again before signing. Open a newly signed report to download an offline, printable HTML copy. Its contents stay tied to the signature even after a later case edit or attachment. Older reports are identified as reconstructions from the available record.

The workbench stores original image bytes, model outputs, individual measurements, annotations and review history in SQLite. The default database is `var/cases.sqlite3`; preserve the configured data directory across restarts. Existing cases retain their original results. A manually archived historic case may still require review.

### Acceptance and deployment

```bash
.venv/bin/python -m pip install pytest httpx
.venv/bin/python -m pytest -q backend/tests
pnpm install --frozen-lockfile
node scripts/smoke-visits.mjs
```

The Chrome test uses an isolated case database. Set `CHROME_BIN` and `PYTHON` to the browser and Python environment when needed. For a separately hosted service, configure `SPORT_MODEL_DATA_DIR` on persistent storage, HTTPS, non-demo credentials and `SPORT_MODEL_SECURE_COOKIE=1`. A custom single-user installation uses `SPORT_MODEL_USERNAME`, `SPORT_MODEL_PASSWORD` and `SPORT_MODEL_DISPLAY_NAME`. The server hosting the workbench must stay online for other users to reach it.

## Models and evaluation

| Component | Operating rule | Verified result |
|---|---|---|
| DINOv2 Small whole-film screening | The deployed head was fitted on all 84 films. A score of at least 0.1788344393 flags a film for review. | Separate matched five-fold models correctly classified 82/84 films, missing 2/68 positives with 0/16 false alarms. |
| Retrained YOLO11s localization | Runs on a 640 × 640 input with rotation-corrected training boxes. Candidate box threshold is 0.20. | In the historical holdout, it located 3/12 annotated lesions and produced two false boxes at this threshold. |
| CPU deployment | The production DINO preprocessing and ONNX model were compared with the original PyTorch pipeline. | All 25 reference decisions agreed. The maximum absolute score difference was below 0.00000009. |

The two outputs can disagree. For example, the workbench flagged a left-fore frontal film on whole-film screening even though YOLO returned no candidate boxes. Reported scores are model outputs, and signed findings remain the reviewer's responsibility. The internal dataset is small and acquisition conditions may correlate with labels. In a separate control, film height alone classified 81/84 correctly. See the research guide for padding experiments, complete per-film predictions and reproduction commands. [Image-study guide](image-study/README.md) · [Five-fold comparison](image-study/published/cv84/three-model-summary.json) · [Training history](training/README.md)

## Browser demo and repository layout

| Path | Contents |
|---|---|
| `backend/` | FastAPI, SQLite, the case viewer, report workflow and CPU inference |
| `app/`, `lib/`, `public/` | Static Next.js site, image processing, browser-side ONNX runtime and models |
| `image-study/`, `training/` | Whole-film experiments, detection training, evaluations and reproducibility records |
| `.github/workflows/deploy.yml` | Static-site build and GitHub Pages deployment |

From the repository root, run `pnpm install --frozen-lockfile`, then `pnpm build`. Serve `out/` with a static HTTP server. For a development server use `pnpm dev`; for the browser smoke test use `node scripts/smoke.mjs`. On the macOS configuration requiring the WebAssembly SWC fallback, set `NEXT_TEST_WASM_DIR` to `$PWD/node_modules/@next/swc-wasm-nodejs` when building.

For a project site, set `NEXT_PUBLIC_BASE_PATH` to the repository name prefixed with a slash; a user site uses an empty base path. The Pages workflow sets this during its build. Enable GitHub Actions as the Pages source in the repository settings. The browser downloads the selected model and ONNX WebAssembly runtime. The fast choice is YOLO11n (about 10 MB); the larger choice is YOLO11s (about 36 MB). The browser letterboxes the image, runs detection, suppresses overlapping boxes and maps coordinates back to the original. Clearing browser storage also clears the local history. [Static-site workflow](.github/workflows/deploy.yml) · [Browser code](app/page.tsx)
