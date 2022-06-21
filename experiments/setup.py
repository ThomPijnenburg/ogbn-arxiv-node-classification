from pathlib import Path

from nodeclf.data import load_dataset
from nodeclf.data import load_paper_title_abstract


data_path = "data/raw/"

load_dataset(data_path=data_path)
load_paper_title_abstract(data_path=data_path)
