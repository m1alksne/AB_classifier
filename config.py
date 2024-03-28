from pathlib import Path

repo_path = Path(__file__).parent

# Path to Data:
# if inside repository
# xwavs_path = repo_path / 'labeled_data' / 'xwavs'

# if absolute path is needed:
xwavs_path = Path('/srv/starter_content/_Shared-Storage_/AB_classifier/wav_files')