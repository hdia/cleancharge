\# CleanCharge
CleanCharge is a lightweight toolkit for analysing electric vehicle (EV) charging costs and emissions using open electricity data.
It provides scripts to fetch data, analyse system-wide and per-origin charging behaviour, and generate reproducible plots.

\## Repository structure
cleancharge/
├── src/
│ ├── analyse/ # Analysis scripts
│ ├── fetch/ # Data fetching and preprocessing
│ └── plots/ # Plotting scripts
├── requirements.txt
├── run\_from\_existing\_data.ps1
└── (data/, results/ created locally – ignored by git)

\## Quick start
\*\*Clone the repo\*\*
```bash
git clone https://github.com/hdia/cleancharge.git
cd cleancharge

\*\*Set up Python environment\*\*
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt

\*\*Run with existing data\*\*
.\\run\_from\_existing\_data.ps1

\*\*Run specific scripts\*\*
python src/analyse/ev\_charging\_analyser\_system.py --help
python src/plots/plot\_map\_per\_origin.py --help

\## Input data
- Not included in repo: .venv/, data/, and results/ are excluded to keep the repository lightweight.
- Supply your own data/processed/ files (e.g., openelectricity\_90d\_hybrid\_local\_with\_intensity.csv, landmarks.csv).

\## Outputs
Scripts write to:
- data/processed/ev\_outputs/ (summary CSVs and plots)
- results/figures/ (maps and other figures)

\## Dependencies
- All dependencies are pinned in requirements.txt.
- Tested on Python 3.11+.

\## Contributing
- Open issues or PRs for bugs, docs, or new modules.

\## License
This project is licensed under the MIT License – see the LICENSE file for details.

\## Acknowledgements
This work was developed as part of research at Swinburne University of Technology on analysing electric vehicle (EV) charging costs and emissions using open electricity data in Melbourne, Australia.



