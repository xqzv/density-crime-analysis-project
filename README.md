# Geospatial Data Analysis

![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://geospatial-data-analysis.streamlit.app/)

Comparative analysis of NYPD and LAPD arrest patterns (2010–2019), exploring relationships between population density and crime arrest rates across New York City and Los Angeles.

## Overview

Urban density and policing strategies are inextricably linked, yet comparing distinct metropolitan models like New York and Los Angeles is difficult due to fragmented data standards.

**Geospatial Data Analysis** unifies over a decade of arrest records from NYPD and LAPD into a standardized, interactive intelligence platform. It enables policy makers, journalists, and researchers to:

*   **Audit Enforcement**: Compare arrest rates normalized by population density across disparate urban environments.
*   **Identify Bias**: Visualize demographic disparities and enforcement patterns across different geographies.
*   **Optimize Policy**: Distinguish between seasonal crime spikes and systemic enforcement trends to inform resource allocation.

## Platform Capabilities

- **Unified Data Schema**: Harmonizes disparate NYPD and LAPD datasets into a single analytical model, standardizing age, gender, race, and offense taxonomies.
- **Interactive Intelligence**: Streamlit-powered dashboard enabling real-time exploration of multi-year trends with granular demographic filtering.
- **Geospatial & Temporal Analysis**: Advanced visualization suite (Plotly/Folium) for identifying localized hotspots and cyclical enforcement patterns.
- **Reproducible Pipeline**: Modular, production-ready architecture designed for extensibility and transparent data lineage.

## Repository Structure

```text
geospatial-data-analysis/
├── app/
│   └── streamlit_app.py       # Interactive Streamlit dashboard
├── data/
│   ├── lapd_aligned.csv       # Processed LAPD data
│   ├── nypd_aligned.csv       # Processed NYPD data
│   ├── sample_lapd.csv        # Sample LAPD data for testing
│   └── sample_nypd.csv        # Sample NYPD data for testing
├── docs/
│   └── Project-Plan.md        # Development roadmap and documentation
├── notebooks/
│   └── report.ipynb           # Narrative analysis notebook
├── scripts/
│   └── create_samples.py      # Utility script to create data samples
├── src/
│   ├── data_processing.py     # Data alignment and transformation logic
│   ├── io.py                  # Data loading and saving utilities
│   ├── standardize.py         # Column mapping and cleaning functions
│   └── visualization.py       # Plotting and visualization functions
├── tests/
│   └── test_processing.py     # Unit tests for data processing
├── main.ipynb                 # Main analysis notebook
├── pyproject.toml             # Project configuration and dependencies
└── README.md                  # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd geospatial-data-analysis
    ```

2.  **Install dependencies:**

    Using `uv` (Recommended):
    ```bash
    uv sync
    ```

    Using `pip`:
    ```bash
    pip install .
    ```

    To install visualization dependencies:
    ```bash
    pip install ".[viz]"
    ```

## Usage

### Interactive Dashboard
Access the live application here: [**Geospatial Data Analysis Dashboard**](https://geospatial-data-analysis.streamlit.app/)

Or launch it locally to explore the data:
```bash
streamlit run app/streamlit_app.py
```

### Analysis Notebook
Open `main.ipynb` or `notebooks/report.ipynb` in VS Code or Jupyter Lab to view the detailed analysis and code.

### Running Tests
Execute the test suite to ensure data processing logic is correct:
```bash
pytest
```

## Data Sources

The analysis is based on publicly available arrest records:

1.  **NYPD Arrests Data (Historic)**: [NYC Open Data](https://data.cityofnewyork.us/Public-Safety/NYPD-Arrests-Data-Historic-/8h9b-rp9u/about_data)
2.  **LAPD Arrest Data (2010-2019)**: [Los Angeles Open Data](https://data.lacity.org/Public-Safety/Arrest-Data-from-2010-to-2019/yru6-6re4/about_data)

*Note: The `data/` directory contains aligned and standardized versions of these datasets.*

## Methodology

To enable direct comparison, the raw data undergoes a rigorous standardization process:
*   **Age**: Mapped to standard groups (<18, 18-24, 25-44, 45-64, 65+).
*   **Gender**: Standardized to Male, Female, Unknown.
*   **Race/Ethnicity**: Mapped to common categories (Black, White, Hispanic, Asian/Pacific Islander, etc.).
*   **Offenses**: Categorized into broad groups (Felony, Misdemeanor, Violation).

## Ethics & Transparency

This project analyzes publicly available law enforcement data. Analysis focuses on *arrest patterns* and should not be interpreted as causal claims about crime or demographic factors. Arrest data reflects enforcement decisions, not actual crime rates, and carries inherent biases in policing practices.

**Key Caveats**:
- Arrest data does not represent crime victimization.
- Disparities in arrests may reflect policing allocation, not underlying criminal activity.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

