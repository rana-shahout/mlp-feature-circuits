"""Isolated administrative identity indexing. Never loads labels or any model."""
import numpy as np
from .core import keys,sha,write
def index(path,out):
 with np.load(path) as z:identities=keys(z['a'])
 np.save(out/'protected_final_keys.npy',identities)
 write(out/'reservation_access.json',{'access':'a only; g not loaded; no model, predictions, raw examples or population summaries','source':str(path),'source_sha256':sha(path),'index_sha256':sha(out/'protected_final_keys.npy')})
