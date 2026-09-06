# Strategic Customer Segmentation Report

## Executive Summary

The analysis used K-Means clustering to group 200 customers using four characteristics: Age,
Gender, Annual Income, and Spending Score. The final model produced five clusters with
different average customer profiles.

Three findings are most useful for deciding what to test next:

**1. Income alone does not explain the difference in Spending Score.** Two higher-income
clusters have similar average incomes but very different Spending Scores: Cluster 3 ($82.1K
income, 54.4 Spending Score) versus Cluster 1 ($85.2K income, 14.1 Spending Score).

**2. Cluster 2 has the highest average Spending Score** (70.2, across 42 customers averaging
age 28.7 and $60.9K income), making it a useful starting population for testing whether
differentiated messaging or offers produce a measurable response.

**3. The clusters describe the sample. Their commercial value still needs to be tested.** The
dataset contains no revenue, profit, campaign response, customer lifetime value, purchase
frequency, or product preference data. The five clusters provide testable customer groups,
not finished marketing personas or proven strategies.

**Management implication:** use the clearest segment contrasts to design controlled marketing
tests, then judge the segmentation by whether those groups actually respond differently.

## Business Problem & Analytical Approach

Customers with similar incomes can show very different spending profiles. The dataset
contains 200 customer records with four attributes available for segmentation: Age, Annual
Income, Spending Score, and Gender. Spending Score is a 1-100 index assigned by the mall,
rather than a monetary spending value.

**Business question:** Can multiple customer characteristics identify groups worth treating
differently in a marketing experiment?

The analysis progressed through three stages:

1. **Simple clustering:** income used by itself, to understand the basic K-Means workflow.
2. **Two-variable clustering:** Annual Income + Spending Score, to examine visible customer
   differences.
3. **Final multivariate model:** Age + encoded Gender + Annual Income + Spending Score.

## Segmentation Methodology

The final segmentation uses Age, Annual Income, Spending Score, and encoded Gender. Gender
was converted into a numeric dummy variable using `pd.get_dummies`. The four inputs were
then standardized using scikit-learn's `StandardScaler`, which centers each feature's values
around its mean and scales them relative to its standard deviation, preventing a
larger-scale feature such as annual income from disproportionately influencing the
Euclidean-distance calculations K-Means relies on.

K-Means was tested across multiple values of K. The multivariate elbow curve was used to
select **K = 5** for the final model. Five clusters should be treated as a practical model
choice for this 200-customer sample: the elbow method does not establish that five permanent
customer types exist in a broader population.

## Segment Portfolio

| Cluster | Customers | Avg. age | Avg. income | Avg. Spending Score | Working description |
|---|---:|---:|---:|---:|---|
| 2 | 42 | 28.7 | $60.9K | 70.2 | Younger, mid-income, highest Spending Score |
| 4 | 38 | 27.3 | $38.8K | 56.2 | Younger, lower-income, mid-high Spending Score |
| 3 | 49 | 37.9 | $82.1K | 54.4 | Higher-income, mid-range Spending Score |
| 0 | 51 | 56.5 | $46.1K | 39.3 | Older, moderate-income, lower Spending Score |
| 1 | 20 | 39.5 | $85.2K | 14.1 | Higher-income, lowest Spending Score |

These are descriptive labels based on observed means, not persona names or inferred
motivations.

## Key Customer Findings

The strongest contrast appears between two higher-income groups: Cluster 3 ($82.1K income,
54.4 Spending Score) and Cluster 1 ($85.2K income, 14.1 Spending Score). Their average
incomes differ by only about $3.1K, while their average Spending Scores differ by more than
40 points. Income alone would place these customers relatively close together; the
multivariate segmentation separates groups whose observed profiles differ substantially on
Spending Score. The dataset cannot explain the cause of that difference: additional
behavioral data would be required to determine whether product preference, visit frequency,
purchase history, customer tenure, or another factor explains the contrast.

Cluster 2 (42 customers, highest average Spending Score of 70.2, average income $60.9K)
provides a clear population for an initial test: useful for testing whether a differentiated
treatment produces a measurable change in behavior, but this does not establish that Cluster
2 generates the most revenue or profit.

## Strategic Recommendations & Measurement Plan

**1. Start with the clearest segment contrasts.** Use Cluster 2 and Cluster 1 as initial
testing populations.

**2. Test treatments before assigning permanent personas.** Use defined messages, offers, or
experiences and measure whether customer response differs across segments (response rate,
conversion, visit frequency, basket value): these metrics are not present in the current
dataset and would need to be collected.

**3. Add behavioral variables to future segmentation.** A stronger customer model could
incorporate transaction history, recency, purchase frequency, product/category behavior,
and campaign response.

**4. Validate the cluster structure before operational use.** Re-run the segmentation on a
larger customer population, using additional validation techniques beyond the elbow method
and checking whether the cluster profiles remain stable.

**Management conclusion:** the five clusters provide hypotheses for customer strategy.
Measured customer behavior determines whether they become useful business segments.

## Appendix A: Model Methodology & Definitions

- **Analysis population:** 200 customer records, each with CustomerID, Gender, Age, Annual
  Income (k$), and Spending Score (1-100). The final labeled dataset also preserves
  intermediate cluster assignments and the final `Multivariate_Cluster` value.
- **Spending Score** is a 1-100 index contained in the source dataset, not a dollar amount. A
  difference of ten Spending Score points represents a difference in the source index, not a
  known dollar difference in customer spending.
- **Final model inputs:** Age, encoded Gender, Annual Income, Spending Score.
- **Gender encoding:** the categorical Gender variable was converted into a numeric dummy
  variable using `pd.get_dummies` before inclusion in the final model.
- **Feature standardization:** the four final inputs were transformed using scikit-learn's
  `StandardScaler` to prevent the raw numerical scale of one feature from dominating the
  Euclidean-distance calculation used by K-Means.
- **K-Means** partitions observations into a selected number of clusters by iteratively
  assigning observations to cluster centers and updating those centers. The final analysis
  uses K = 5.
- **Model selection:** the elbow method was used to inspect within-cluster variation across
  candidate values of K. The multivariate elbow curve supported five clusters as the working
  model. The elbow method was the primary cluster-count diagnostic in this project: no
  silhouette-score or cluster-stability analysis was used as part of the final selection.

## Appendix B: Supporting Analysis & Interpretation Limits

The analysis progressed through income-only clustering, then Annual Income + Spending Score,
then the final Age + Gender + Income + Spending Score model. The earlier models were
exploratory steps and should not be confused with the final five-cluster multivariate
segmentation.

**Score-to-income index:** the notebook calculates Spending Score / Annual Income, preserved
in the final CSV as `Spend_Income_Ratio`. Cluster 4 had the highest average value reported
during the project: **1.85**. This is best described as an exploratory score-to-income index.
Spending Score is an index rather than dollars spent, so this value does not represent a
percentage of income spent, financial efficiency, profitability, or customer value.

**What the source data does not contain:** revenue, profit, customer lifetime value, campaign
response, purchase frequency, basket value, product preference, or reason for Spending Score.
The clustering cannot directly support claims about those outcomes.

**Segment labels:** cluster numbers are model identifiers. This report uses descriptive
summaries based on observed means, such as "Cluster 1: Higher-income, lowest Spending Score",
rather than behavioral persona names, to avoid assigning motivations or commercial value that
the dataset does not measure.

**Validation boundary:** the segmentation describes this 200-customer sample. Before
operational use, a stronger validation process would include a larger customer population,
additional behavioral variables, more than one cluster-quality measure, cluster-stability
testing, and measured customer-response outcomes.
