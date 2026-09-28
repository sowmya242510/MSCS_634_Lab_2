# MSCS 634 – Lab 2: Classification Using KNN and RNN

**Name:** Sowmya Cheruku  
**Course:** Advanced Big Data and Data Mining (MSCS 634)  
**University:** University of the Cumberlands  
**Instructor:** Satish Penmatsa

## Purpose
In this lab I compared two distance-based classifiers, K-Nearest Neighbors (KNN) and Radius Neighbors (RNN), on the Wine dataset from scikit-learn (178 samples, 13 chemical features, 3 wine classes). I split the data 80/20 into training and test sets, tried different values of `k` (1, 5, 11, 15, 21) and `radius` (350–600), and plotted how test accuracy changed with each parameter.

## Key Insights
| Model | Parameter | Accuracy |
|---|---|---|
| KNN | k = 1 / 5 / 11 / 15 / 21 | 0.778 / 0.722 / 0.750 / 0.750 / 0.778 |
| RNN | radius = 350 | 0.750 |
| RNN | radius = 400–600 | 0.722 (flat) |

- KNN did slightly better than RNN (best 77.8% vs. 75.0%) and was less sensitive to its parameter.
- KNN accuracy went up and down with `k` without a clear trend. The test set only has 36 samples, so one prediction is about 2.8%, and I think most of the differences are noise.
- RNN accuracy went flat once the radius reached 400. At that size nearly every training point is inside the radius, so the prediction is basically the overall class mix.
- Both models scored lower than I expected. The reason is that the features are on very different scales (`proline` has a standard deviation of ~314, most others are under 1), so distance is dominated by one feature.
- As an extra check, I standardized the features and re-ran KNN. Accuracy rose to about 94–97% for all k values, which supports this explanation.
- KNN is a safer default. RNN can be useful when the distance scale is meaningful and you want to flag points with no close neighbors, but it is harder to tune.

## Challenges and Decisions
- **Radius values:** 350–600 looked oddly large at first. The median distance between points in the unscaled data is around 300, so these values are reasonable only because of the unscaled `proline` feature.
- **Empty neighborhoods:** `RadiusNeighborsClassifier` throws an error if a test point has no neighbors within the radius, so I set `outlier_label='most_frequent'`.
- **Scaling:** The lab did not ask for scaling, so the main results use the raw data as instructed. I added the scaling comparison at the end as a bonus.
- **Reproducibility:** I used `random_state=42` for the train/test split. A different split would change the exact accuracies.

## Files
- `MSCS_634_Lab_2.ipynb` – the notebook with code, plots and discussion
- `README.md` – this file
