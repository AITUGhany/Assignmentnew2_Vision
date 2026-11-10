# A2 - Generative data augmentation for food classification

4-class food classifier (bread / drinks / fruit / snacks). Extends the A1 baseline CNN with
extra real photos (Food-101), synthetic images from **TA-TiTok + MaskGen**, a **ConvNeXt-Tiny**
classifier, Grad-CAM, nearest-neighbour retrieval and a Gradio web app.

> Status: ArtAug (LoRA for FLUX.1-dev) was **not run** (does not fit on a free Colab T4), so the
> ablation has two variants: `real` and `real + synthetic`. All models reach 100 % on the
> test set, so the test set cannot show a benefit of synthetic data. See the report.

## Repository layout

```
Ghani_FoodClassification_A2.ipynb   main notebook (data, training, evaluation, app)
generate_synthetic.ipynb            TA-TiTok + MaskGen image generation (Untitled1.ipynb)
app.py                              standalone local version of the web app
requirements.txt
results/                            tables, figures, metrics.json  (models: see link below)
report/A2_Report_FoodClassification.pdf
.gitignore
```

Model weights (`convnext_tiny_best.keras`, `baseline_cnn.keras`): **<ADD LINK, e.g. Google Drive>**
Dataset (`dataset.zip`, own photos): not in the repository - **<ADD LINK or say: available on request>**

## Environment

Tested in Google Colab: Linux 6.6 x86_64, Python 3.13, TensorFlow 2.21.0, NVIDIA T4 GPU.

## Reproduce in Colab (GPU = T4)

1. **Synthetic images.** Open `generate_synthetic.ipynb`, run it; it clones
   `bytedance/1d-tokenizer`, generates 100 images per class into `synthetic/raw/<class>/`
   and you zip them as `synthetic.zip`.
2. **Archives on Google Drive** (My Drive root): `dataset.zip`, `extra_real.zip`
   (folders `bread/ drinks/ fruit/ snacks/`; `drinks` and `fruit` may be empty), `synthetic.zip`
   (contains `synthetic/raw/<class>/`, optionally `synthetic/artaug/<class>/`).
3. Open the main notebook, then run in this order:
   mount Drive -> copy the three archives to `/content` (and unzip `extra_real.zip`) ->
   imports/unzip cell -> data loading -> dataset table -> definitions -> A1 baseline ->
   ConvNeXt experiments -> comparison -> Grad-CAM -> kNN -> latency -> save results -> app.
   Skip the Food-101 download cells if `extra_real.zip` already exists.
4. Results are written to `/content/results` and downloaded as `results.zip`.

Check after the dataset-table cell: `extra real: True | synthetic raw: True`.
Synthetic images are capped at 30 % of the training set and are never used for validation/test.

## Run the app locally

```bash
pip install -r requirements.txt
# expected folders next to app.py:
#   convnext_tiny_best.keras          (from the results / weights link)
#   dataset/train/<class>/*.jpg       (own training photos, used for nearest neighbours)
#   synthetic/raw/<class>/*.png       (optional, for "Generate similar")
python app.py
```

Open the printed local URL. `app.py` is a standalone port of the notebook's app cell; if a
layer name differs in your saved model, the script prints the available layers.

## Ethics

Synthetic images are AI-generated, always labelled SYNTHETIC in the app, and kept out of
validation/test. Do not upload photos of people without their consent; images are processed
in memory and not stored.

## References

- ByteDance 1d-tokenizer (TA-TiTok, MaskGen): https://github.com/bytedance/1d-tokenizer
- ArtAug: arXiv:2412.12888, ModelScope DiffSynth-Studio
- ConvNeXt (Liu et al., CVPR 2022), Grad-CAM (Selvaraju et al., ICCV 2017), Food-101 (Bossard et al., ECCV 2014)
