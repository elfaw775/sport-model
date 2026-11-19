# Image-level comparison plan

Outcome is one binary label per film, positive for an annotated lesion. The archived manifest supplies the image IDs, labels and splits. It has 48 positive and 11 negative films in training, 10 and 3 in validation, and 10 and 2 in the historical test set.

## Model selection

- Two frozen Hugging Face encoders are candidates for the primary model: DINOv2 Small and ResNet18. Each probe first normalizes each embedding to unit length, standardizes each dimension using training data, and fits an L2 logistic regression with C=1 and balanced class weights.
- Fit five stratified, shuffled folds on training films alone (seed 42), refitting the scaler and classifier within each fold. Rank candidates by pooled out-of-fold ROC-AUC, then average precision, and prefer DINOv2 on an exact tie. Fit the winning probe again on all 59 training films. Pooling here is a model-selection summary, not a test estimate. The folds are film-level because a verifiable horse-ID table is unavailable.
- Separately compare archived YOLO11s, retrained YOLO11s, a lesion-supervised patch classifier, an image-acquisition control and the always-positive rule. For YOLO the film score is the maximum saved lesion-box confidence; zero if the archived predictor returned no box. The acquisition score is the absolute distance from the median negative training-film height.
- Tune each nonconstant decision threshold on the same 13 validation films only. Call scores at or above the threshold positive. Compare observed scores and a threshold above the maximum. Maximize balanced accuracy, then sensitivity, then specificity, then the numeric threshold. Do not use validation or test to choose between encoders.
- Freeze model selection, fitted probe, validation scores, and all operating thresholds before producing candidate predictions for the test split. Score only the winning HF probe and the five predefined comparators on the historical test set. The test split has already been used in prior experiments.

## Output

Report TP, FP, TN, FN, plain accuracy, sensitivity, specificity, balanced accuracy, precision, ROC-AUC and average precision. Two-sided exact binomial 95% intervals cover accuracy, sensitivity, specificity and precision where their denominators exist. The intervals are descriptive and assume independent films. A constant predictor can reach 10/12 accuracy by prevalence alone. Neither a high accuracy on this test set nor an advantage over YOLO establishes generalization to newly acquired X-rays. Examine acquisition artifacts separately.

## Amendment before historical test scoring

After viewing the train-only out-of-fold results for the original pair, we added RAD-DINO as a separate exploratory comparison. It receives the same fixed probe and validation threshold rule, and cannot replace the selected primary model. If its checkpoint is unavailable when freezing, record the omission. Where available, score this single named secondary along with the locked primary and original comparators.
