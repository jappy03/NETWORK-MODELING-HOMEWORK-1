# Homework 1: Source modeling

Falasco Jacopo - Student ID 2194529

This experiment models an outgoing Google Meet transport using measured batch
inter-arrival times, batch sizes and individual packet workloads. The notebook
follows the complete analysis, from the original CSV to a synthetic trace and
comparison with the measurements.

## Download the experiment

1. On this repository page, click **Code > Download ZIP**.
2. Extract the downloaded ZIP on your computer.
3. Keep the filenames unchanged. The files needed for Colab are
   `MeetAnalysis_Final.ipynb` and `meet_outgoing.csv`.

The notebook already includes outputs from a complete run. All figures and
tables are also supplied in `results_final/`, alongside the original input files.

## Files

| File or folder | Purpose |
| --- | --- |
| [MeetAnalysis_Final.ipynb](MeetAnalysis_Final.ipynb) | Main notebook, with numbered sections and short English comments |
| [meet_outgoing.csv](meet_outgoing.csv) | Original exported data: 39,083 selected outgoing packets |
| [meet5min.pcapng](meet5min.pcapng) | Original approximately five-minute Wireshark capture |
| [MeetAnalysis_Final.py](MeetAnalysis_Final.py) | Equivalent Python script for local execution |
| [results_final/](results_final/) | Generated data, parameter tables, numerical comparisons and PNG/PDF figures |
| [requirements.txt](requirements.txt) | Python package versions used for the verified run |
| [packet_selection.txt](packet_selection.txt) | Wireshark display filter for the selected flow and time window |
| [transport_verification.json](transport_verification.json) | Summary of the selected WebRTC transport |
| [checksums.sha256](checksums.sha256) | SHA-256 checksums of the distributed files |

## Run step by step in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Choose **File > Upload notebook** and select `MeetAnalysis_Final.ipynb`
   from the extracted folder. If an opening dialog appears first, use its
   **Upload** tab. Sign in to your Google account if requested.
3. Connect to a Python runtime. A standard CPU runtime is sufficient.
4. Run the first code cell using its play button. When it requests a file,
   upload only `meet_outgoing.csv` from the same extracted folder.
5. Continue through the remaining cells from top to bottom. Each section
   displays the values, tables or plots needed for the next part of the analysis.
   Alternatively, select **Runtime > Run all**, then choose the CSV when prompted.
6. The last cell creates `results_final/` and `MeetAnalysis_Final_Results.zip`.
   Open the Files panel on the left, find the ZIP, and use its menu to download it.

The PCAPNG does not need to be uploaded to Colab: the notebook analyzes the
supplied CSV. The original capture is included so that packet selection can be
checked independently in Wireshark. Reopening a Colab notebook in a new runtime
may require uploading the CSV again.

## Notebook order

| Section | Operation | Main output |
| --- | --- | --- |
| 1. Load the data | Read the original CSV and define helpers | File name, row count and sample rows |
| 2. Measure the traffic | Compute packet statistics and form batches | Packet summary, batch statistics and empirical batch probabilities |
| 3. Plot measured distributions | Plot histograms, the PMF and empirical CDFs | `01_Empirical_Distributions` |
| 4. Estimate the models | Fit marginal laws and simple reference distributions | Workload-group estimates, parameter table and CDF fit distances |
| 5. Generate the synthetic trace | Sample the source with seed 42 | Measured/synthetic statistics, relative differences and CDF distances |
| 6. Compare distributions | Overlay measured, model and synthetic distributions | `02_Final_Distributions_With_Models` and `03_Final_CDF_QQ_Comparison` |
| 7. Compare one-second rates | Aggregate both traces over complete windows | Rate comparison table and `04_Final_Traffic_Rate_Comparison` |
| 8. Save the results | Export data, tables, plots and metadata | `results_final/` and the downloadable results ZIP |

## Measurement definitions

The selected outgoing UDP transport is
`192.168.1.7:61626 -> 74.125.250.248:3478`. Its endpoints were cross-checked
with the final call's WebRTC selected candidate pair. Associated control traffic
on that transport is included. Other flows in the raw capture are excluded.

The original capture time window is [60, 299.974147] s, giving an observation
duration of 239.974147 s after the initial 60 s are removed. Successive packet
gaps of at most 0.5 ms are grouped into the same batch. This consecutive-gap rule
does not identify video frames. It yields 15,883 batches and 15,882 intervals.

- `A_i`: difference between successive batch start times, in seconds.
- `B_i`: number of packets in batch i.
- `W_i,k`: individual IP `Total Length` multiplied by eight, in bits.

Packet-level inter-arrival statistics are displayed before batching and differ
from the batch inter-arrival statistics used for modeling `A_i`. In both cases,
the interval count is one less than the number of packet or batch timestamps.

## Models and parameter estimation

- Inter-arrivals: Gamma(rate lambda_A, shape k_A), estimated from the sample mean
  and sample variance.
- Batch sizes: `B_i = 1 + Z_i * (3 + X_i)`, where Z is Bernoulli and X is Poisson,
  estimated from the measured batch-size mean and variance. The constants reflect
  the main measured peaks at one and four packets.
- Individual workloads: two normal groups, separated at 5,000 bits inside the
  observed gap. Group frequencies, means and sample standard deviations estimate
  the proposal mixture. Nonpositive proposals are rejected by redrawing both the
  component and value, giving a mixture conditioned globally on positive values.

The source samples these variables independently across batches and assigns
the same timestamp to all packets in a synthetic batch. Exponential, geometric,
shifted Poisson, single normal and Gamma alternatives remain simple comparison
laws in the fitting table.

## Expected results

| Quantity | Measured | Synthetic |
| --- | ---: | ---: |
| Batch count | 15,883 | 15,909 |
| Packet count | 39,083 | 39,141 |
| Mean batch interval [ms] | 15.108858 | 15.083332 |
| Mean batch size [packets] | 2.460681 | 2.460305 |
| Mean individual workload [bits] | 6910.024512 | 6973.933874 |
| Whole-window IP rate [Mbit/s] | 1.125390 | 1.137484 |
| Standard deviation of complete 1 s rates [Mbit/s] | 0.071377 | 0.134232 |

The measured-to-model CDF distances are 0.06312 for A, 0.02167 for B and 0.07325
for W. The whole-window average rate differs by about +1.07%. However, the
one-second rate standard deviation is about 88.06% higher in the synthetic
trace, with a rate CDF distance of 0.26778. There are 239 complete one-second
windows; the final partial window is excluded from these distribution statistics.

The main comparison tables are `final_synthetic_statistics.csv`,
`final_synthetic_cdf_distances.csv` and `final_traffic_rate_comparison.csv` in
`results_final/`. Exact fitted parameters are in `model_parameters.csv`.

## Scope and limitations

The same short call is used for fitting, model selection and comparison. These
are descriptive diagnostics, without an independent validation set or formal
hypothesis tests. The batch model cannot produce sizes 2 and 3, which together
account for 4.04% of measured batches. It also underrepresents rare large batches.
The continuous workload model smooths discrete packet sizes and is not a
generator of valid IP packets. The independent source discards temporal and
cross-variable dependence, while simultaneous synthetic packets remove measured
within-batch delays. Marginal agreement and a close mean rate therefore do not
ensure that aggregate traffic variability is reproduced accurately.

## Software versions and local execution

`requirements.txt` records the versions used to generate the included results.
Colab's default package versions may differ. To use the recorded versions in
Colab, upload `requirements.txt` before running the notebook and execute
`%pip install -q -r requirements.txt` in a separate setup cell. If Colab requests
a runtime restart, restart it, upload the CSV again when requested, and run the
notebook from the beginning. The notebook records the versions actually used in
`results_final/analysis_metadata.json`.

For local execution, keep the CSV beside the script and run from this directory:

```bash
python -m pip install -r requirements.txt
python MeetAnalysis_Final.py
```

The verification environment uses Python 3.12.14. A fresh runtime starts
all calculations from the CSV; the notebook does not load saved model parameters
or previously generated result tables.

## Check the selection in Wireshark

1. Open `meet5min.pcapng` in Wireshark.
2. Apply the display filter from `packet_selection.txt`.
3. Display time relative to capture start in seconds, preserving timestamp precision.
4. Add a custom `Total Length` column for the IPv4 field `ip.len`.
5. Export the displayed packet dissections as CSV. The original columns are
   `No.`, `Time`, `Source`, `Destination`, `Protocol`, `Length`, `Total Length`, `Info`.
   The selection contains 39,083 packets. Use IP `Total Length`, not frame `Length`.

The supplied CSV is the original export, unchanged. The raw capture also contains
other flows; only the filtered transport and time window enter this analysis.
