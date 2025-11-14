# CleanCharge

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**CleanCharge** is a lightweight toolkit for analysing electric vehicle (EV) charging costs and emissions using open electricity data. It provides scripts to fetch data, analyse system-wide and per-origin charging behaviour, and generate reproducible plots.

---

## Repository structure

```
cleancharge/
├── src/
│   ├── analyse/   # Analysis scripts
│   ├── fetch/     # Data fetching and preprocessing
│   └── plots/     # Plotting scripts
├── requirements.txt
├── run_from_existing_data.ps1
└── (data/, results/ created locally – ignored by git)
```

---

## Quick start

### 1. Clone the repo
```bash
git clone https://github.com/hdia/cleancharge.git
cd cleancharge
```

### 2. Set up Python environment
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Run with existing data
```powershell
.
un_from_existing_data.ps1
```

### 4. Run specific scripts
```bash
python src/analyse/ev_charging_analyser_system.py --help
python src/plots/plot_map_per_origin.py --help
```

---

## Data inputs

The CleanCharge workflow requires two authoritative input datasets:

- `openelectricity_90d_hybrid_local_with_intensity.csv`
- `openelectricity_emissions_30d_local.csv`

These have been pre-processed from the OpenElectricity API and represent the authoritative 90-day and 30-day datasets used in the CleanCharge analysis.

👉 Download them from Zenodo:  
[https://doi.org/10.5281/zenodo.17232110](https://doi.org/10.5281/zenodo.17232110)

Place both files into:
```
data/processed/
```

The helper script `run_from_existing_data.ps1` assumes this location. Once the files are in place, you can reproduce the analyses and regenerate figures by following the steps above.

---

## Outputs

Scripts write to:
- `data/processed/ev_outputs/` (summary CSVs and plots)  
- `results/figures/` (maps and other figures)

---

## Dependencies

- All dependencies are pinned in `requirements.txt`.  
- Tested on **Python 3.11+**.

---

## Contributing

Open issues or PRs for bugs, docs, or new modules.

---

## License

[MIT License](LICENSE)

---

## Cite this work

If you use CleanCharge in your research, please cite both the dataset and the code:

- **Dataset:**  
  Dia, H. (2025). *CleanCharge: Authoritative processed electricity datasets (30-day and 90-day).* Zenodo.  
  [https://doi.org/10.5281/zenodo.17232110](https://doi.org/10.5281/zenodo.17232110)

- **Code:**  
  Dia, H. (2025). *CleanCharge: Analysis and forecasting scripts (v1.0).* Zenodo.  
  [https://doi.org/10.5281/zenodo.17232338](https://doi.org/10.5281/zenodo.17232338)
