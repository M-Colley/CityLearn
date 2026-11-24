"""Write the weather columns used by the BOPTEST comparison for the recorded periods (days 8-20 and 99-112)."""
import sys
import pandas as pd

source, target = sys.argv[1:3]
w = pd.read_csv(source, usecols=['time', 'TDryBul', 'HDirNor', 'HDifHor'])
keep = ((w.time >= 8 * 86400) & (w.time <= 20 * 86400)) | ((w.time >= 99 * 86400) & (w.time <= 112 * 86400))
w[keep].to_csv(target, index=False)
