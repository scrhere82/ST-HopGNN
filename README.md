ST-HopGNN LTE implementation
Research code for ST-HopGNN (recurrent ESN, thresholded multi-hop graph aggregation) and LSTM, TCN, STGCN, AGCRN, DCRNN, and MTGNN baselines.
Setup
Install Python, PyTorch, PyTorch Geometric, NumPy, pandas, scikit-learn, matplotlib, and Jupyter. Use versions compatible with your hardware.
Data (not distributed)
Place authorized CSV telemetry sequences in data/train/, data/validate/, data/test/. Each file contains a time/index column followed by 19 sensor columns in the same order. The notebook fits a MinMaxScaler on training sequences only. Do not upload restricted telemetry.

Set STHOPGNN_DATA_DIR to change the input folder, and STHOPGNN_RESULTS_DIR to change the output folder. Defaults: ./data, ./results.
Reproduction
Open ST_HopGNN_LTE_GitHub_Clean.ipynb and run cells sequentially. The notebook uses seed 42, window 5, horizon 3, four hops, adjacency threshold 0.5, and adjacency freezing after 20 completed epochs. Checkpoint selection uses minimum validation loss, which may occur before adjacency freezing.

No data, trained weights, predictions, or run outputs are included. Results may depend on environment and data access.
