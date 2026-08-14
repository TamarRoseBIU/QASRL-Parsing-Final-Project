# Custom Trainer with Custom Loss Function


This README provides an overview of the custom loss function and its implementation stages.

## Custom Loss Function Overview

The custom loss function operates in the following stages:

1. **Prediction Parsing**:
   - For each prediction, split the content into individual QAs using the `<QA>` token.

2. **Question and Answer Separation**:
   - For each QA, separate the question from its corresponding list of answers using the `?` token.

3. **Answer Tokenization**:
   - Split the list of answers for each question into individual answers using the `<A>` token.

4. **Target Parsing**:
   - Repeat steps 1–3 for the target labels to process the ground truth data.

5. **List Alignment**:
   - Equalize the lengths of the prediction and target answer lists by padding the shorter list with "empty answers" (empty lists). This ensures missing answers are taken into account during loss calculation.

6. **Similarity Scoring**:
   - Compute the Intersection over Union (IoU) scores for all answers in the prediction list against all answers in the target list. These scores are used as the similarity metric.

7. **Score Matrix Construction**:
   - Build a score matrix of size `number_of_answers × number_of_answers`, where the cell at position `(i, j)` represents the IoU score between the `i`-th predicted answer and the `j`-th target answer.

8. **Answer Alignment**:
   - Use the Hungarian algorithm (via `scipy.optimize.linear_sum_assignment`) to align the prediction and target answers. This ensures the optimal pairing of predicted and target answers based on similarity.

9. **Loss Calculation**:
   - Calculate the Cross-Entropy (CE) loss for each pair of aligned answers (prediction and target).

10. **Final Loss Computation**:
    - Sum the CE losses for all aligned pairs.
    - Calculate `LAMBDA1 * CE_loss + LAMBDA2 * custom_loss`, where `LAMBDA1` and `LAMBDA2` are hyperparameters, to produce the final loss value.

## Performance Optimization

Initially, the loss calculation was implemented naively by iterating over each example in the batch and calculating the loss for each aligned pair of answers.
To improve performance, the implementation was optimized to allow the CE loss function to process the entire batch of predictions and target answers in a single call.
This batch-wise computation significantly reduces runtime overhead.

