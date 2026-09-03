# Data Directory Details

Because the raw dataset exceeds GitHub's 50 MB recommended file limit, raw files are tracked via `.gitignore` and must be downloaded via the instructions below.

## Dataset Overview
- **Name:** Used Cars Dataset (Austin Reese Craigslist Scrape)
- **Source URL:** https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data
- **Size:** ~1.45 GB uncompressed (`vehicles.csv`), 426,880 rows × 26 columns

## How to Obtain
Execute the download snippet in Python:
```python
import kagglehub
import shutil
import os

path = kagglehub.dataset_download("austinreese/craigslist-carstrucks-data")
os.makedirs("data/raw", exist_ok=True)
shutil.copy(f"{path}/vehicles.csv", "data/raw/vehicles.csv")