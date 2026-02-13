# MTI-EVO Replication Kit
        
## Purpose
Independent verification of the "Chimera Effect" and noise resistance claims.

## Quick Start
1. Install: `pip install -r requirements.txt`
2. Run validation: `python scripts/run_validation.py`
3. View results: `jupyter notebook analysis/results.ipynb`

## Expected Results
- Noise tolerance ratio: > 2.0× (signal vs noise)
- Flow maintenance rate: > 90%
- Hallucination prevention: > 90%

## File Structure
/data/raw_logs - Original experiment logs
/data/brain_dumps - Serialized .mti-brain states
/code/core - MTI-EVO source code
/code/tests - Validation test suite
/analysis - Jupyter notebooks for analysis

## Contact
For questions: mediataginteractive@gmail.com
