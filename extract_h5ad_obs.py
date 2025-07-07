#!/usr/bin/python

import sys
import scanpy


ts_file = scanpy.read_h5ad(sys.argv[1])

ts_file.obs.to_csv(sys.argv[2], sep=',', index=False)
