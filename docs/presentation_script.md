# RecycleVision: live evaluation script

The evaluation is 30 minutes:

| Part | Time |
|---|---|
| Self-introduction | 3 min |
| Project explanation | 12 min |
| Mock questions | 10 min |
| Feedback | 5 min |

It is scored out of 60: code quality, documentation, modularity, presentation, task accomplishment and the mock
questions are worth 10 each. The explanation below follows GUVI's **"Model Building Projects"** template, in order.
Keep each point to 1–2 lines and always say *why*.

**Have open on screen:** the live app, the GitHub README, and `reports/RecycleVision_Report.pdf`.

---

## 1. Domain (3 lines)
Waste management is a huge cost for cities, and a large share of recyclables ends up in landfill because it is
mis-sorted. Manual sorting is slow, unhygienic and error-prone. Computer vision can sort waste from a photo: in smart
bins, on conveyor lines, or in an app that teaches people which bin to use.

## 2. Problem statement (1 line)
Classify a photo of a waste item into its category (10 classes, including plastic, metal, glass, paper, cardboard and
organic) and serve it in a Streamlit app with a confidence score.

## 3. Data cleaning and preprocessing
- **I used all three datasets the brief links.** That is 30,301 images in total.
- **The key finding: they overlap heavily.** 11,232 images were duplicates; for example, 71% of the V2 dataset's images
  are copies of images in the other two sets. If I had concatenated them, the same photo would sit in both train and test, and the
  test accuracy would be fake.
- **How I de-duplicated:**
  - Exact copies are found with a SHA-1 hash of the file bytes.
  - Near copies (resized or re-encoded) need a perceptual-hash distance ≤ 8 *and* a thumbnail correlation ≥ 0.95.
  - I calibrated the second check by looking at borderline pairs. A simple pixel difference merged *different* bottles
    on white backgrounds, and missed the *same* flyer under different lighting.
- **Other cleaning:**
  - Removed 7 unreadable, tiny or blank images.
  - Removed 20 photos whose copies had different labels.
  - Merged brown, green and white glass into one glass class; renamed biological to organic.
- **Every image's fate** is recorded in `data/manifest.csv`, so the dataset rebuilds identically on any machine.
- **Preprocessing:** resized to 224×224, the ImageNet input size. Normalisation is built *inside* each model, so the
  app can't get it wrong.

## 4. EDA (1–2 lines of findings)
- **Clothes** was about a third of the raw data, so I capped every class at 2,500.
- **Image widths** ranged from 68 to 7,786 px, which is why everything is resized.
- **Colour:** battery and trash photos are the brightest (white studio backgrounds), and organic is shifted towards
  red and yellow. Background colour is a shortcut the model could learn, so I added colour and brightness augmentation
  and checked with Grad-CAM that the model looks at the object, not the background.

## 5. Feature engineering
- **No hand-made features.** CNNs learn their features; ImageNet pre-training provides generic edge and texture filters.
- **Data augmentation is the "feature engineering" for images:**
  - flips and rotation (up to ±72°), because waste lies at any angle;
  - zoom and shift;
  - contrast and brightness, for lighting changes.
- Augmentation runs inside the model during training only.

## 6. Statistical significance
- **McNemar's exact test** compares two classifiers on the *same* test images, using only the images where exactly one
  of the two is right. EfficientNetB0 versus each other model gives p < 0.001, so its advantage isn't chance.
- **A bootstrap 95% confidence interval** for macro-F1 (1,000 resamples) gives 0.955–0.971.

## 7. Class imbalance
- **Cap:** at most 2,500 images per class.
- **Class weights:** "balanced" weights in the loss, so small classes like battery count as much as large ones.
- **Metric:** macro-F1 for model selection, which averages the classes equally, so a model can't win by being good
  only at clothes.

## 8. Base model and why
- **A CNN from scratch** (4 convolution blocks) as the baseline, to measure what transfer learning adds.
- **Result:** 56.3% accuracy. With about 11k training images, a network learning from zero can't match one
  pre-trained on 1.2 million images.

## 9. Models used and why
- **Transfer learning with ImageNet weights:**
  - **MobileNetV2:** light and fast, built for mobile and edge devices like a smart bin.
  - **EfficientNetB0:** best accuracy per parameter, thanks to compound scaling.
  - **ResNet50:** the classic, deeper and heavier reference.
- **Two-phase training:**
  1. **Phase 1:** freeze the backbone and train only the new head (global average pooling → dropout → Dense 256 →
     softmax over 10 classes) at learning rate 1e-3.
  2. **Phase 2:** unfreeze the top 35–45% of the backbone, keeping BatchNorm frozen, and fine-tune at learning rate
     1e-4 with cosine decay. Fine-tuning adapts high-level features to waste textures; BatchNorm stays frozen so its
     statistics aren't destroyed by small batches.
- **Training details:**
  - Early stopping restores the best epoch.
  - The optimiser is AdamW.
  - Trained on a free Colab T4 GPU with mixed precision, then exported to float32 for CPU deployment.

## 10. Evaluation metric and why
- **Accuracy** is the brief's target (> 85%).
- **Macro-F1 is the selection metric**, because it balances precision and recall *per class* regardless of class size.
- Also reported: precision, recall, top-3 accuracy, false-positive rate and confusion matrices.

## 11. Final model and why
- **EfficientNetB0**, chosen by the highest **validation** macro-F1. The test set was never used for choosing.
- **On the 2,423 test images:**
  - 96.5% accuracy and macro-F1 0.963;
  - precision 0.962 and recall 0.965, which are balanced;
  - top-3 accuracy 99.7% and a false-positive rate of 0.40%.
- **It is also deployment-friendly:** 18 MB, with about 18 ms of model compute per image on a laptop CPU (about 0.3 s per upload in the
  app, including Streamlit overhead). ResNet50 is 97 MB and less
  accurate.

## 12. Conclusion and "feature importance"
- **Grad-CAM is the image equivalent of feature importance.** It highlights where the model looked. On correct
  predictions it focuses on the object: the bottle, the can, the orange.
- **Most confusions are physically plausible:** glass → plastic (13 images), cardboard ↔ paper, glass → metal.
- **Several of the "most confident mistakes" are label errors in the dataset:** a metal lid labelled glass, sneakers
  labelled clothes. The model is right there, so the true accuracy is slightly higher than reported.

## 13. Business suggestions
- **Use the confidence score:** below 80% the app says "not sure". I chose 80% on the validation set: it flags about
  5% of images but catches half of all mistakes, so accuracy on the rest rises to 97.5%. In a plant, those flagged
  items go to a human sorter.
- **Fine-tune on real bin-camera photos:** most training photos are clean, single-object shots.
- **Battery = hazardous alert:** batteries must never go in household recycling. Battery recall is 100%.
- **Use MobileNetV2 on cheap edge hardware:** it's 11 MB and about 12 ms per image, if a 2-point accuracy drop is
  acceptable.

---

## Demo flow (inside the 12 minutes, about 2 minutes)
1. Open the app and pick the **plastic** example: 100% plastic, top-3 shown, Grad-CAM on the bottle.
2. Upload your own photo of something at hand (a bottle, a can, a paper cup).
3. Open the **Model comparison** tab: the table and confusion matrix.

## Likely mock questions (1–2 per technology)
- **Why did you merge three datasets, and why de-duplicate?**
  More and more varied data, but the sets copy each other. Without de-duplication, test images leak into training.
- **What is transfer learning? Why freeze, then fine-tune?**
  Reuse ImageNet features. Training only the head first stops large random-head gradients from wrecking the
  pre-trained weights; then a low learning rate adapts the top layers.
- **Why keep BatchNorm frozen while fine-tuning?**
  Its running statistics come from ImageNet. Updating them on small batches makes them noisy and hurts accuracy.
- **What does data augmentation do? Is it applied at test time?**
  It creates new variations of the training images to reduce overfitting. No: the Keras augmentation layers are only
  active during training.
- **Precision vs recall vs F1? Why macro rather than weighted?**
  - Precision: of the items predicted plastic, how many really are plastic.
  - Recall: of the real plastic items, how many were caught.
  - F1: their harmonic mean.
  - Macro weights every class equally, so the small classes matter.
- **How did you handle overfitting?**
  Augmentation, dropout, weight decay (AdamW), early stopping on validation loss, and a separate validation set.
- **Why EfficientNetB0 and not ResNet50?**
  Higher F1, five times smaller, and about 2x faster on CPU (18 vs 38 ms). Compound scaling balances depth, width and resolution.
- **What is Grad-CAM, and why use logits?**
  The gradient of the class score with respect to the last conv feature maps weights those maps into a heatmap.
  Softmax saturates near 100% confidence, so its gradients vanish; the logit doesn't.
- **What is McNemar's test?**
  A paired test on the disagreements between two classifiers over the same test images.
- **What is mixed precision?**
  float16 compute with float32 weights. It's about twice as fast on T4 tensor cores; the final layer stays float32
  for numerical stability.
- **How does the Streamlit app work? What is `st.cache_resource`?**
  It loads the model once per server, not on every click.
- **What would you do with more time?**
  - Clean the label errors.
  - Fine-tune on real bin-camera photos.
  - Add object detection for images with several items.
  - Quantise to TFLite for edge devices.
- **Python:** what is a generator? A decorator? Why did you build a package (`src/recyclevision`)?
  A package gives reusable, importable modules shared by the scripts, the app and the tests.

## Self-introduction (3 min): yours to write
Your name, education, current role, why data science and AI, and a one-line pitch for this project.
