from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent

def haber() -> pd.DataFrame:
  """
  Tabularized counts of cell types in the 
  small intestinal epithelium of mice with different conditions.
  
  Haber et al. 2017
 
  Returns
  -------
  data matrix as pandas data frame.
    
  """
  filename = HERE / 'haber_counts.csv'
  
  return pd.read_csv(filename)
