# Ambulatory glucose profile analysis tool

## DISCLAIMER ##

This tool is for research and educational purposes only. It is NOT a medical device and has NOT been validated for clinical diagnosis or treatment decisions.

Do not adjust medications, change diet, or make health decisions based solely on this output. Exercise caution and always consult a qualified healthcare professional for interpretation of glucose data and any treatment adjustments.

### Overview ###

This tool generates a comprehensive Ambulatory Glucose Profile (AGP) with extended clinical metrics, including Time in Tight Range (TITR) metric. It processes continuous glucose monitoring (CGM) data and produces both a visual AGP plot and detailed statistical analysis.

### Features ###

#### General  ####

- Full AGP visualization with percentile curves (5-95%, IQR)
- Rate of Change (ROC) profile
- Circadian binning for time-of-day patterns

#### Time in Range metrics with level breakdowns: ####

- TIR (70-180 mg/dL)
- TITR (70-140 mg/dL) - Tight target
- TAR with Level 1 (181-250) and Level 2 (>250)
- TBR with Level 1 (54-69) and Level 2 (<54)

#### Advanced variability metrics: ####

- MAGE (Mean Amplitude of Glycemic Excursions)
- MODD (Mean of Daily Differences)
- CONGA (Continuous Overall Net Glycemic Action)

#### Risk indices: ####

- LBGI (Low Blood Glucose Index)
- HBGI (High Blood Glucose Index)
- ADRR (Average Daily Risk Range)

### AUC analysis: ####

- total, above range, below range

### Data quality assessment: ####

- wear time, reading frequency

### Requirements ###

#### Packages ####

`pip install -r requirements.txt`

#### Input Data Format ####

- Excel file with two columns:
- Time: datetime (e.g., "2024-01-01 08:00:00")
- Sensor Reading(mg/dL): numeric glucose values

### Usage ###

bash

`python get_sensor.py your_glucose_data.xlsx`


## Example ##

![Example](files/ambulatory_glucose_profile.png)

### Metrics Explained ###

#### Core Metrics ####

Metric	Description	Clinical Target

- TIR	Time in Range (70-180 mg/dL)	≥70%
- TITR	Time in Tight Range (70-140 mg/dL)	≥50%
- TBR	Time Below Range (<70 mg/dL)	<4%
- CV	Coefficient of Variation	<36%
- GMI	Glucose Management Indicator	below 7%
- J-Index	Combined mean + variability	n/a

#### Variability Metrics ####

Metric	Description

- MAGE	Mean amplitude of glycemic excursions
- MODD	Day-to-day glucose variability
- CONGA	Intra-day glycemic variability (1h lag)

### Risk Metrics ####

Metric	Description

- LBGI	Low Blood Glucose Index
- HBGI	High Blood Glucose Index
- GRI	Glycemia risk index
- ADRR	Average Daily Risk Range

#### Data Quality ####

Metric	Description	Warning Threshold

- Readings/day	Average daily readings	<24 readings/day
- Wear time	% of possible readings	<70%
- Severe hypo/week	Events <40 mg/dL per week	n/a

### Configuration ###

Key parameters at script top:

- LOW = 70			# Lower bound standard range
- HIGH = 180			# Upper bound standard range
- TIGHT_LOW = 70		# Lower bound tight range
- TIGHT_HIGH = 140		# Upper bound tight range
- BIN_MINUTES = 5		# Time bin size for AGP
- ROC_CLIP = 10		# Rate of change physiological limit

### Limitations & Warnings ###

- Minimum data: AGP typically requires ≥5 days for reliability
- Data gaps: Long gaps (>2 hours) may affect MODD and ADRR calculations
- Sensor accuracy: Assumes CGM-grade data; fingerstick data may have limitations
- MAGE calculation: Uses smoothed data; may differ from manual calculation

### Interpretation Tips ###

- TITR >50% suggests acceptable glycemic control
- TBR >4% indicates need for hypoglycemia prevention
- CV >36% suggests unstable glucose, consider variability-reducing therapy
- Nighttime patterns: Shaded area helps identify nocturnal hypoglycemia; however, a sharp "V-shaped" drop with a rapid recovery may indicate sensor compression artifact rather than true hypoglycemia
- ROC spikes indicate rapid changes; correlate with meals/exercise
- GMI < 7.0%: Suggests acceptable glycemic control (Note: GMI estimates A1c from CGM data, but may differ from lab A1c).

### References ###

#### Ambulatory Glucose Profile (AGP) ####

1. **Mazze RS, Strock E, Wesley DM, et al.** (1987). Characterizing diabetes control with 'The ambulatory glucose profile'. *Journal of Diabetes and its Complications*, 1(2):260-267. [PubMed](https://pubmed.ncbi.nlm.nih.gov/3333689/)

2. **Battelino T, Danne T, Bergenstal RM, et al.** (2019). Clinical Targets for Continuous Glucose Monitoring Data Interpretation: Recommendations From the International Consensus on Time in Range. *Diabetes Care*, 42(8):1593-1603. [doi:10.2337/dci19-0028](https://diabetesjournals.org/care/article/42/8/1593/36345/Clinical-Targets-for-Continuous-Glucose)

#### Time in Range Metrics (TIR, TBR, TAR, TITR) ####

3. **Battelino T, et al.** (2019). Clinical Targets for Continuous Glucose Monitoring Data Interpretation: Recommendations From the International Consensus on Time in Range. *Diabetes Care*, 42(8):1593-1603. [Link](https://diabetesjournals.org/care/article/42/8/1593/36345/Clinical-Targets-for-Continuous-Glucose)

#### Coefficient of Variation (CV) ####

4. **Monnier L, et al.** (2008). Glycemic Variability: Should We and Can We Prevent It? *Diabetes Care*, 31 Suppl 2:S150-4. [Link](https://diabetesjournals.org/care/article/31/Supplement_2/S150/25288)

#### Glucose Management Indicator (GMI) ####

5. **Bergenstal RM, et al.** (2018). Glucose Management Indicator (GMI): A New Term for Estimating A1C From Continuous Glucose Monitoring. *Diabetes Care*, 41(11):2275-2280. [Link](https://diabetesjournals.org/care/article/41/11/2275/36534)

#### J-Index ####

6. **Wojcicki JM.** (1995). J-Index. A new proposition of the assessment of current glucose control in diabetic patients. *Hormone and Metabolic Research*, 27(1):41-42. [PubMed](https://pubmed.ncbi.nlm.nih.gov/7710282/)

#### MAGE (Mean Amplitude of Glycemic Excursions) ####

7. **Service FJ, Molnar GD, Rosevear JW, Ackerman E, Gatewood LC, Taylor WF.** (1970). Mean amplitude of glycemic excursions, a measure of diabetic instability. *Diabetes*, 19(9):644-655. [PubMed](https://pubmed.ncbi.nlm.nih.gov/5476146/)

#### MODD (Mean of Daily Differences) ####

8. **Molnar GD, Taylor WF, Ho MM.** (1970). Day-to-day variation of continuously monitored glycaemia: a further measure of diabetic instability. *Diabetologia*, 6(4):342-347. [Link](https://link.springer.com/article/10.1007/BF01228243)

#### CONGA (Continuous Overall Net Glycemic Action) ####

9. **McDonnell CM, Donath SM, Vidmar SI, Werther GA, Cameron FJ.** (2005). A novel approach to continuous glucose analysis utilizing glycemic variation. *Diabetes Technology & Therapeutics*, 7(2):253-263. [PubMed](https://pubmed.ncbi.nlm.nih.gov/15738710/)

#### LBGI & HBGI (Low/High Blood Glucose Index) ####

10. **Kovatchev BP, et al.** (2001). Assessment of risk for severe hypoglycemia among patients with type 1 and type 2 diabetes using self-monitoring blood glucose data. *Diabetes Care*, 24(11):1870-1875. [Link](https://diabetesjournals.org/care/article/24/11/1870/22463/Assessment-of-Risk-for-Severe-Hypoglycemia-Among)

#### ADRR (Average Daily Risk Range) ####

11. **Kovatchev BP, et al.** (2006). The average daily risk range: a new measure of glycemic variability. *Diabetes Care*, 29(11):2272-2277. [Link](https://diabetesjournals.org/care/article/29/11/2272/24793/The-Average-Daily-Risk-Range-a-New-Measure-of)

#### GRI (Glycemia Risk Index) ####

12. **Hill NR, et al.** (2011). A novel approach to assessing glycemic variability in type 1 diabetes: the glycemic risk index. *Diabetes Technology & Therapeutics*, 13(8):835-841. [Link](https://www.liebertpub.com/doi/10.1089/dia.2011.0041)

### License ###

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.