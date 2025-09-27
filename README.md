# LLM_PT_Alerts (TEST)

**LLM demonstrator using GTFS-Realtime public transport service alerts**

The demonstrator processes GTFS-Realtime alerts (train, tram, and bus) and applies large language models (LLMs) to:

- Parse and normalise route identifiers  
- Generate plain-language passenger summaries  
- Translate alerts into Mandarin and Arabic  
- Compute run-level metrics (alerts processed, mode counts, route resolution rate)  

## Repository structure

- `scripts/` – Python scripts for preprocessing and summarisation  
- `config/` – Sample GTFS reference files and configuration templates  
- `data/` – Example input alerts and sample outputs (non-sensitive, anonymised)  
- `results/` – Tables and figures generated from demonstrator runs  
- `requirements.txt` – Python package dependencies  

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/hdia/LLM_PT_Alerts.git
cd LLM_PT_Alerts
pip install -r requirements.txt
```

## Quickstart (using sample data)

Python 3.9+ is recommended.

```bash
# Run summarisers
python scripts/summarise_runs.py "data/sample_alerts/mel_*.csv"
python scripts/summarise_runs.py "data/sample_alerts/seq_*.csv"
python scripts/summarise_runs_syd.py "data/sample_alerts/syd_*.csv"

# Validate outputs
python scripts/validate_outputs.py \
  results/tables/_runs_summary_MEL.csv \
  results/tables/_runs_summary_SYD.csv \
  results/tables/_runs_summary_SEQ.csv

# Compute cross-city averages
python scripts/compute_averages.py --settings config/settings.yaml
```

The `data/sample_alerts/` folder contains small anonymised CSVs that demonstrate the workflow without requiring full GTFS datasets. Full datasets are excluded from this repository.

## Citation

If you use this repository, please cite:

```bibtex
@misc{dia2025llmptalerts,
  author       = {Hussein Dia},
  title        = {LLM_PT_Alerts: LLM Demonstrator using GTFS-Realtime Public Transport Service Alerts},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.xxxxxxx},  # update once DOI is minted
  url          = {https://doi.org/10.5281/zenodo.xxxxxxx}
}
```


## License

This project is licensed under the MIT License – see the LICENSE
 file for details.

## Acknowledgements

This work was developed as part of research on LLM-assisted multilingual disruption alerts in public transport.
