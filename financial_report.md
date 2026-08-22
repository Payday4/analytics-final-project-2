
# Readmission Prevention Financial Impact Report

## Executive Summary
- Test set size: 20,340
- Baseline readmissions: 2,265
- No-intervention baseline cost: $33,975,000
- Best-performing model: **Fine-tuned XGBoost**
- Estimated total savings vs no intervention: **$1,761,000**
- Mean savings per patient: **$87**
- 95% confidence interval for mean savings per patient: **$55 to $120**
- Permutation p-value: **0.0000**
- One-sample t-test p-value: **0.0000**

## Financial Comparison
| Model | Intervention Cost | Readmission Cost | Total Cost | Savings vs No Intervention | ROI |
| --- | --- | --- | --- | --- | --- |
| Base Logistic Regression | $20,223,000 | $15,405,000 | $35,628,000 | $-1,653,000 | -8.17% |
| Base XGBoost | $162,000 | $33,510,000 | $33,672,000 | $303,000 | 187.04% |
| Fine-tuned XGBoost | $8,439,000 | $23,775,000 | $32,214,000 | $1,761,000 | 20.87% |

## Statistical Significance
| Model | Mean Savings / Patient | 95% CI Low | 95% CI High | Permutation P-value | One-sample t-test P-value |
| --- | --- | --- | --- | --- | --- |
| Fine-tuned XGBoost | $87 | $55 | $120 | 0.0000 | 0.0000 |
| Base XGBoost | $15 | $9 | $22 | 0.0000 | 0.0000 |
| Base Logistic Regression | $-81 | $-127 | $-33 | 0.0008 | 0.0005 |

## Interpretation
- A 95% CI that does not include zero suggests the savings are unlikely to be random noise.
- A permutation p-value below 0.05 indicates the average savings are statistically significant.
- If the model shows both positive savings and a significant p-value, the intervention appears to produce real financial value.
