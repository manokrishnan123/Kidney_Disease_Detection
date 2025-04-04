import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "cnnClassifier"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"{project_name}/logger.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/utils/common.py",
    f"{project_name}/config/__init__.py",
    f"{project_name}/config/configuration.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "templates/index.html",
    "research/trials.ipynb"
]

for filepath in list_of_files:
    filepath = Path(filepath)  # Handling / from List of files to read the folder properly 
    filedir, filename = os.path.split(filepath) # return folder and filename separately
    
    
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir} for the file: {filename}")
        
    if (not os.path.exists(filename)) or (os.path.getsize(filepath) == 0): # check if the file exists and ignore if something is written in the file
        with open(filepath, 'w') as file:
            pass
            logging.info(f"Created an empty file: {filename}")
            
    else:
        logging.info(f"File already exists: {filename}")