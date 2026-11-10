"""Standalone local version of the A2 web app (port of the notebook's Gradio cell).
Not tested outside Colab - check the layer names printed on start if something fails."""
import os, random, time, platform
import numpy as np, tensorflow as tf, gradio as gr
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)
CLASSES = ["bread", "drinks", "fruit", "snacks"]
MODEL_PATH = "convnext_tiny_best.keras"
TRAIN_DIR, SYN_DIR = "dataset/train", "synthetic/raw"
EXT = (".jpg", ".jpeg", ".png", ".webp")

model = tf.keras.models.load_model(MODEL_PATH)
clf = model.get_layer("classifier")
base = next((l for l in model.layers if l.name.startswith("convnext")), None)
if base is None:
    print("Layers:", [l.name for l in model.layers]); raise SystemExit("ConvNeXt backbone layer not found")
embed_model = tf.keras.Model(base.input, tf.keras.layers.GlobalAveragePooling2D()(base.output))

def load_img(p):
    return np.array(tf.keras.utils.load_img(p, target_size=IMG_SIZE), dtype=np.float32)

# gallery of real training photos
paths, labels = [], []
for i, c in enumerate(CLASSES):
    d = os.path.join(TRAIN_DIR, c)
    for f in sorted(os.listdir(d)):
        if f.lower().endswith(EXT): paths.append(os.path.join(d, f)); labels.append(i)
def embed(batch):
    v = embed_model.predict(batch, verbose=0); return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
gallery = np.concatenate([embed(np.stack([load_img(p) for p in paths[i:i+32]])) for i in range(0, len(paths), 32)])

def gradcam(img):
    x = tf.convert_to_tensor(img[None], tf.float32)
    with tf.GradientTape() as tape:
        fmap = base(x, training=False); tape.watch(fmap)
        logits = tf.matmul(tf.reduce_mean(fmap, axis=(1, 2)), clf.kernel) + clf.bias
        score = tf.gather(logits[0], tf.argmax(logits[0]))
    g = tape.gradient(score, fmap)
    cam = tf.nn.relu(tf.reduce_sum(tf.reduce_mean(g, axis=(1, 2), keepdims=True) * fmap, axis=-1))[0]
    return (cam / (tf.reduce_max(cam) + 1e-8)).numpy(), tf.nn.softmax(logits[0]).numpy()

def overlay(img, cam, a=0.45):
    up = tf.image.resize(cam[..., None], IMG_SIZE).numpy()[..., 0]
    return np.clip((1 - a) * img / 255.0 + a * plt.cm.jet(up)[..., :3], 0, 1)

lat = []
def classify(pil):
    if pil is None: return None, None, [], "Upload or capture a food photo.", ""
    t0 = time.perf_counter()
    img = np.array(pil.convert("RGB").resize(IMG_SIZE), dtype=np.float32)
    cam, probs = gradcam(img)
    q = embed(img[None])[0]; nn = np.argsort(-(gallery @ q))[:5]
    ms = (time.perf_counter() - t0) * 1000; lat.append(ms)
    info = (f"**Latency:** {ms:.0f} ms (avg {np.mean(lat):.0f} ms over {len(lat)})  \n"
            f"**Environment:** {platform.platform()} | TF {tf.__version__} | "
            f"GPU: {[g.name for g in tf.config.list_physical_devices('GPU')] or 'none (CPU)'}")
    gal = [(Image.open(paths[i]).convert("RGB"), f"{CLASSES[labels[i]]} ({(gallery[i] @ q):.2f})") for i in nn]
    return ({c: float(probs[i]) for i, c in enumerate(CLASSES)},
            Image.fromarray((overlay(img, cam) * 255).astype("uint8")), gal, info, CLASSES[int(probs.argmax())])

def stamp(im):
    im = im.convert("RGB").resize((256, 256)); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 256, 22], fill=(200, 0, 0)); d.text((6, 5), "SYNTHETIC / AI-GENERATED", fill=(255, 255, 255)); return im

def similar(cls):
    d = os.path.join(SYN_DIR, cls) if cls else ""
    if not cls or not os.path.isdir(d): return [], "No synthetic images found for this class."
    fs = [f for f in os.listdir(d) if f.lower().endswith(EXT)]
    return [(stamp(Image.open(os.path.join(d, f))), "synthetic") for f in random.sample(fs, min(4, len(fs)))], \
           "These images are AI-generated (TA-TiTok + MaskGen, generated offline) and are not real photos."

with gr.Blocks(title="Food classifier - A2") as demo:
    gr.Markdown("# Food classifier (bread / drinks / fruit / snacks)\nDo **not** upload photos of people without consent; images are processed in memory and not stored. SYNTHETIC = AI-generated.")
    with gr.Row():
        inp = gr.Image(type="pil", sources=["upload", "webcam"], label="Food photo")
        with gr.Column():
            lbl = gr.Label(num_top_classes=4, label="Prediction"); cam_out = gr.Image(label="Grad-CAM")
    info_md = gr.Markdown(); nn_gal = gr.Gallery(label="Nearest real photos", columns=5, height=230)
    state = gr.State(""); btn = gr.Button("Generate similar (synthetic examples)")
    syn_gal = gr.Gallery(label="SYNTHETIC examples (AI-generated)", columns=4, height=260); syn_note = gr.Markdown()
    inp.change(classify, inp, [lbl, cam_out, nn_gal, info_md, state]); btn.click(similar, state, [syn_gal, syn_note])
if __name__ == "__main__":
    demo.launch()
