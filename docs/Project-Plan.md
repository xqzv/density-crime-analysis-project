# Density Crime Analysis — Portfolio Rewrite Plan

## Objective

Rework the academic notebook into a reproducible, modular, and polished portfolio project that highlights data engineering, visualization, and deployment skills for employers.

---

## High-Level Approach

1. Make the project reproducible and lightweight for reviewers (requirements, sample data, clear README).
2. Refactor exploratory code into a small `src/` package of reusable modules (IO, standardization, processing, visualization).
3. Produce a concise report notebook and an interactive demo (Streamlit) for live exploration.
4. Add tests, CI, Docker, and documentation (Data & Ethics, License).

---

## Key Deliverables

### Phase 1–3 (MVP)

| File | Purpose |
|------|---------|
| `requirements.txt` | Pin key packages |
| `README.md` | Project summary, run instructions, ethics statement |
| `data/sample_nypd.csv`, `data/sample_lapd.csv` | ~5k rows each for demos and CI |
| `src/io.py` | Load/save helpers |
| `src/standardize.py` | Column mapping & cleaning |
| `src/data_processing.py` | Align/create datasets, sampling |
| `src/visualization.py` | Static and interactive plot functions |
| `notebooks/report.ipynb` | Clean, narrative-first notebook with final figures |
| `tests/test_data_processing.py` | 8–10 focused pytest tests |
| `LICENSE` | MIT |

### Phase 4 (Deployment)

| File | Purpose |
|------|---------|
| `app/streamlit_app.py` | Interactive demo: year slider, sample fraction, view selector; deployed to Streamlit Cloud |

### Phase 5+ (Polish, optional)

| File | Purpose |
|------|---------|
| `.github/workflows/ci.yml` | Run tests and lint |
| `run.ps1` | One-command local test/lint runner |

---

## Recommended Technologies

| Category | Tools | Rationale |
|----------|-------|-----------|
| **Core** | Python, pandas, numpy | Standard data stack |
| **Visuals** | Plotly, Folium, matplotlib/seaborn | Interactive charts, maps, publication-quality PNGs |
| **App** | Streamlit | Quick interactive demos, easy deployment (Streamlit Cloud—free) |
| **Packaging & Testing** | `src/` layout, pytest, GitHub Actions | 8–10 focused tests, CI automation |
| **Reproducibility** | `requirements.txt`, sample CSVs (~5k rows) | Fast CI, easy reviewer setup |

> **Note:** No geopandas or Docker for MVP. Streamlit Cloud deployment prioritized over local Docker.

---

## Execution Plan & Estimated Effort

| Phase | Task | Effort |
|-------|------|--------|
| 1 | **Reproducibility artifacts + sample data** — Create `requirements.txt`, `README.md` (run instructions + ethics section), and two stratified sample CSVs | ≤2h |
| 2 | **Refactor ETL into `src/` and add focused tests** — Move cleaning, renaming, and alignment functions into `src/`; add 8–10 unit tests | 3–6h |
| 3 | **Produce cleaned report notebook** — Lightweight `notebooks/report.ipynb` that imports `src/` and shows final figures with captions | ≤2h |
| 4 | **Build and deploy Streamlit demo** — Interactive controls: year-range slider, sample fraction, view selector. Deploy to Streamlit Cloud | 3–6h |
| 5 | **CI and polish** — Add `run.ps1` (local test/lint runner) and GitHub Actions CI. Refine README and visuals | 1–3h |

---

## Priority Recommendation (MVP-First)

1. **Phase 1:** Add `requirements.txt`, `README.md`, and sample datasets → reviewers can reproduce instantly.
2. **Phase 2:** Refactor ETL into `src/` and write 8–10 focused tests → demonstrates engineering maturity.
3. **Phase 3:** Replace heavy notebook with clean `notebooks/report.ipynb` → shows narrative + final insights.
4. **Phase 4:** Build Streamlit demo and deploy to Streamlit Cloud → live, interactive portfolio piece (big impact).
5. **Phase 5:** Add `run.ps1`, GitHub Actions CI, and final polish → professional finish.

---

## Next Immediate Action

I recommend starting with **Phase 1** (reproducibility + sample data). I can:

- [ ] Generate `requirements.txt` pinned to tested versions
- [ ] Create stratified sample CSVs (5k rows each) from your aligned datasets
- [ ] Draft `README.md` with run instructions and ethics statement

Let me know if you want me to proceed!

---

## Notes

This plan is meant to be pragmatic and staged so you can show incremental improvements in your portfolio.
