# Density Crime Analysis

Comparative analysis of NYPD and LAPD arrest patterns (2010–2019), exploring relationships between population density and crime arrest rates across New York City and Los Angeles.

## Overview

This project analyzes how arrest patterns correlate with population density in two major U.S. cities. It demonstrates data engineering, exploratory data analysis, and interactive visualization skills through a reproducible, modular codebase and interactive Streamlit dashboard.

## Features

- **Data Processing**: Standardized and aligned arrest data from NYPD and LAPD
- **Analysis**: Comparative statistics and density-crime correlation analysis
- **Visualization**: Interactive Plotly charts and Folium maps for geographic exploration
- **Interactive Demo**: Streamlit-powered dashboard for year-range filtering and data exploration

## Getting Started

### Prerequisites

- Python 3.13+
- Dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd density-crime-analysis
   ```

2. Install dependencies:
   Using `pip`:
   ```bash
   pip install .
   ```
   
   Using `uv`:
   ```bash
   uv sync
   ```

3. (Optional) Install visualization dependencies:
   ```bash
   pip install ".[viz]"
   ```

### Running the Project

- **Analysis Notebook**: Open `main.ipynb` to explore the analysis
- **Interactive Dashboard**: `streamlit run app/streamlit_app.py`

## Data Sources

The original raw datasets can be accessed at:

1. [NYPD Arrests Data (Historic)](https://data.cityofnewyork.us/Public-Safety/NYPD-Arrests-Data-Historic-/8h9b-rp9u/about_data)
2. [LAPD Arrest Data (2010-2019)](https://data.lacity.org/Public-Safety/Arrest-Data-from-2010-to-2019/yru6-6re4/about_data)

Processed and aligned datasets are included in the `data/` directory:
- `nypd_aligned.csv` — NYPD arrest records (standardized format)
- `lapd_aligned.csv` — LAPD arrest records (standardized format)

## Project Structure

```
.
├── README.md                    # This file
├── pyproject.toml              # Project metadata and dependencies
├── main.py                      # Entry point
├── CW2 Data Visualization.ipynb # Main analysis notebook
├── data/
│   ├── nypd_aligned.csv        # NYPD arrests data
│   └── lapd_aligned.csv        # LAPD arrests data
├── docs/
│   └── Project-Plan.txt        # Development roadmap
└── src/                        # (In development)
    ├── io.py                   # Data loading/saving utilities
    ├── standardize.py          # Column mapping & cleaning
    ├── data_processing.py      # Alignment & computation
    └── visualization.py        # Plot functions
```

## Technologies

- **Data Processing**: pandas, numpy
- **Visualization**: Plotly, Folium, Matplotlib, Seaborn
- **Web App**: Streamlit
- **Testing**: pytest
- **Environment**: Python 3.13+

## Ethics & Transparency

This project analyzes publicly available law enforcement data. Analysis focuses on *arrest patterns* and should not be interpreted as causal claims about crime or demographic factors. Arrest data reflects enforcement decisions, not actual crime rates, and carries inherent biases in policing practices.

**Key Caveats**:
- Arrest data does not represent crime victimization
- Disparities in arrests may reflect policing allocation, not underlying criminal activity
- Use this analysis for informed discussion, not punitive policy decisions

## Development Roadmap

See `docs/Project-Plan.txt` for the full development plan, including:
- Phase 1: Reproducibility & sample data
- Phase 2: Modular refactoring and testing
- Phase 3: Clean report notebook
- Phase 4: Interactive Streamlit dashboard
- Phase 5: CI/CD and polish

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Questions or Feedback?

For issues, suggestions, or collaboration inquiries, please open an issue or contact the project maintainers.

