# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# The stock loop prints one line per file. With the full pack attached that is
# 3,200+ lines of scrollback, so the paths are collected and summarised instead.
INPUT_FILES = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        INPUT_FILES.append(os.path.join(dirname, filename))

for p in INPUT_FILES[:20]:
    print(p)
if len(INPUT_FILES) > 20:
    print(f"... and {len(INPUT_FILES) - 20} more")
print(f"\n{len(INPUT_FILES)} files under /kaggle/input")

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
