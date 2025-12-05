# Currently Under Reconstruction
___
# Density Crime Analysis

![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://density-crime-analysis-project.streamlit.app/)

Comparative analysis of NYPD and LAPD arrest patterns (2010–2019), exploring relationships between population density and crime arrest rates across New York City and Los Angeles.

## 📖 Overview

This project analyzes how arrest patterns correlate with population density in two major U.S. cities. It demonstrates data engineering, exploratory data analysis, and interactive visualization skills through a reproducible, modular codebase and interactive Streamlit dashboard.

**Key Research Topics:**
*   **Temporal Patterns**: Cyclical variations in criminal activity based on time of day, day of week, month, or year.
*   **Crime Hotspots**: Geographic areas with disproportionately high concentrations of criminal activity.
*   **Enforcement Density**: The concentration of police resources relative to population and geography.

## ✨ Features

- **Data Standardization**: Unified schema for NYPD and LAPD data, standardizing age, gender, race, and offense categories.
- **Interactive Dashboard**: Streamlit-powered application for exploring data with year-range filtering and demographic breakdowns.
- **Advanced Visualization**: Interactive Plotly charts for temporal analysis and Folium maps for geographic exploration.
- **Reproducible Pipeline**: Modular code structure with clear separation of concerns (IO, processing, visualization).

## 📂 Repository Structure

```text
density-crime-analysis/
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

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd density-crime-analysis
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

## 📊 Usage

### Interactive Dashboard
Access the live application here: [**Density Crime Analysis Dashboard**](https://density-crime-analysis-project.streamlit.app/)

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

## 💾 Data Sources

The analysis is based on publicly available arrest records:

1.  **NYPD Arrests Data (Historic)**: [NYC Open Data](https://data.cityofnewyork.us/Public-Safety/NYPD-Arrests-Data-Historic-/8h9b-rp9u/about_data)
2.  **LAPD Arrest Data (2010-2019)**: [Los Angeles Open Data](https://data.lacity.org/Public-Safety/Arrest-Data-from-2010-to-2019/yru6-6re4/about_data)

*Note: The `data/` directory contains aligned and standardized versions of these datasets.*

## 🛠️ Methodology

To enable direct comparison, the raw data undergoes a rigorous standardization process:
*   **Age**: Mapped to standard groups (<18, 18-24, 25-44, 45-64, 65+).
*   **Gender**: Standardized to Male, Female, Unknown.
*   **Race/Ethnicity**: Mapped to common categories (Black, White, Hispanic, Asian/Pacific Islander, etc.).
*   **Offenses**: Categorized into broad groups (Felony, Misdemeanor, Violation).

## ⚖️ Ethics & Transparency

This project analyzes publicly available law enforcement data. Analysis focuses on *arrest patterns* and should not be interpreted as causal claims about crime or demographic factors. Arrest data reflects enforcement decisions, not actual crime rates, and carries inherent biases in policing practices.

**Key Caveats**:
- Arrest data does not represent crime victimization.
- Disparities in arrests may reflect policing allocation, not underlying criminal activity.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

