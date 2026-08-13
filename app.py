import io, random, time, os, shutil
import requests, numpy as np, gradio as gr
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import PIL.ImageFilter as ImageFilter
from PIL import ImageColor
from rembg import remove, new_session
from scipy import ndimage
from scipy.signal import find_peaks

COMFY = "http://127.0.0.1:8188"
W, H, SIZE = 896, 1152, 704
os.makedirs("identity_pack", exist_ok=True)
os.makedirs("brand_kit", exist_ok=True)

SCENES = {
 "Kitchen":  ("empty luxury product photography background, modern kitchen counter made of {surface}, {light}, {styling}, shallow depth of field, professional commercial photograph, horizontal counter surface filling the bottom third of the frame, sharp edge where wall meets counter, camera at counter level, no product",
              "professional product photography, {p} standing upright on a kitchen counter, soft natural contact shadow beneath the product, subtle reflection on the surface, photorealistic, high detail"),
 "Bathroom": ("empty luxury bathroom vanity with a {surface} top, {light}, {styling}, elegant minimal background, professional cosmetics photography, horizontal vanity surface filling the bottom third of the frame, sharp edge where wall meets surface, camera at counter level, no product",
              "professional product photography, {p} standing upright on a bathroom vanity, soft contact shadow, subtle reflection, spa atmosphere, photorealistic, high detail"),
 "Studio":   ("empty seamless studio backdrop in {backdrop}, {light}, professional studio product photography, glossy reflective floor surface filling the bottom third of the frame, clean horizon line, camera at floor level, no product",
              "professional studio product photography, {p} standing upright on a glossy studio surface, soft contact shadow, clean reflection, photorealistic, high detail"),
 "Luxury":   ("empty luxury still life set, {surface}, {light}, {styling}, high-end commercial product photography, elegant minimal composition, horizontal surface filling the bottom third of the frame, sharp edge where backdrop meets surface, camera at surface level, no product",
              "luxury product photography, {p} standing upright on an elegant surface, warm soft shadow, refined reflection, photorealistic, high detail"),
 "Outdoor":  ("empty outdoor setting, {surface}, {light}, {backdrop}, natural lifestyle product photography, horizontal table surface filling the bottom third of the frame, clear separation between ground and surface, camera at table level, no product",
              "lifestyle product photography, {p} standing upright on an outdoor surface, soft natural shadow, gentle reflection, photorealistic, high detail"),
}

VARIANTS = {
 "Kitchen": {
  "surface": ["white marble", "walnut wood", "black granite", "beige quartz"],
  "light": ["soft morning window light", "warm golden hour glow", "bright airy daylight", "moody dusk light"],
  "styling": ["a ceramic bowl of lemons nearby", "fresh herbs in a glass vase", "stacked ceramic dishes in the background", "copper cookware blurred behind"],
 },
 "Bathroom": {
  "surface": ["white marble", "beige travertine", "pale oak wood"],
  "light": ["soft diffused spa lighting", "warm candlelit glow", "bright daylight"],
  "styling": ["rolled white towels", "eucalyptus branches", "lit candles", "a white orchid"],
 },
 "Studio": {
  "backdrop": ["a soft gray gradient", "a warm beige gradient", "a deep charcoal gradient", "a muted sage gradient"],
  "light": ["dramatic rim lighting", "soft diffused lighting", "hard editorial lighting"],
 },
 "Luxury": {
  "surface": ["draped beige silk", "champagne satin", "black glass", "cream stone"],
  "light": ["warm premium lighting", "soft candlelight", "a golden spotlight"],
  "styling": ["subtle gold accents", "a strand of pearls", "a silk ribbon", "dried pampas grass"],
 },
 "Outdoor": {
  "surface": ["a weathered stone table", "a rustic wooden table", "a flat mossy rock"],
  "light": ["golden hour sunlight", "soft overcast light", "dappled tree shade"],
  "backdrop": ["a blurred garden", "forest bokeh", "a lavender field"],
 },
}
NEG = "floating product, missing shadow, text, watermark, person, hand, clutter, low quality, blurry, distorted, bad lighting, cork, stopper, apothecary bottle, bottle inside a bottle, extra glass container, mutated product, changed product shape"

COLOR_NAMES = [("white",(255,255,255)),("black",(25,25,25)),("gray",(128,128,128)),("cream",(245,239,230)),
 ("beige",(213,196,170)),("brown",(139,90,60)),("gold",(200,160,80)),("yellow",(240,210,90)),("orange",(230,140,60)),
 ("red",(200,50,40)),("pink",(240,170,190)),("purple",(140,100,170)),("lavender",(190,170,220)),("blue",(60,110,200)),
 ("navy",(30,50,110)),("teal",(50,150,150)),("green",(90,150,90)),("sage green",(150,170,140)),("olive",(130,140,80))]
def cname(hexc):
    r,g,b = ImageColor.getrgb(hexc)
    return min(COLOR_NAMES, key=lambda n: (n[1][0]-r)**2+(n[1][1]-g)**2+(n[1][2]-b)**2)[0]

print("Loading BiRefNet (first run downloads ~200MB)...")
try:    SESSION = new_session("birefnet-general")
except Exception: SESSION = new_session("isnet-general-use")

def upload(name, pil):
    buf = io.BytesIO(); pil.save(buf, "PNG"); buf.seek(0)
    r = requests.post(f"{COMFY}/upload/image", files={"image": (name, buf, "image/png")}, data={"overwrite": "true"})
    r.raise_for_status(); return r.json().get("name", name)

def run(wf, save_id):
    r = requests.post(f"{COMFY}/prompt", json={"prompt": wf})
    if r.status_code != 200: raise RuntimeError(f"ComfyUI error: {r.text[:200]}")
    pid = r.json()["prompt_id"]
    for _ in range(240):
        time.sleep(1)
        h = requests.get(f"{COMFY}/history/{pid}").json()
        if pid in h and h[pid].get("outputs"):
            img = h[pid]["outputs"][save_id]["images"][0]
            return Image.open(io.BytesIO(requests.get(f"{COMFY}/view",
                   params={"filename": img["filename"], "type": "output"}).content))
    raise TimeoutError("ComfyUI timed out")

CKPT = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "RealVisXL_V5.0_fp16.safetensors"}}

def wf_background(pos1, neg):
    return {"1": CKPT,
     "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pos1}},
     "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": neg}},
     "4": {"class_type": "EmptyLatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
     "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0], "seed": random.randint(1, 2**48), "steps": 25, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1}},
     "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
     "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "panel_bg"}}}

def wf_seat(pos2, anchor_name, alpha_name, neg, denoise=0.25):
    return {"1": CKPT,
     "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": pos2}},
     "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": neg}},
     "4": {"class_type": "LoadImage", "inputs": {"image": anchor_name}},
     "5": {"class_type": "LoadImage", "inputs": {"image": alpha_name}},
     "6": {"class_type": "ImageToMask", "inputs": {"image": ["5", 0], "channel": "red"}},
     "7": {"class_type": "InvertMask", "inputs": {"mask": ["6", 0]}},
     "8": {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["1", 2]}},
     "9": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["8", 0], "mask": ["7", 0]}},
     "10": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["9", 0], "seed": random.randint(1, 2**48), "steps": 20, "cfg": 6.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": denoise}},
     "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["1", 2]}},
     "12": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": "panel_seat"}}}

def prepare_product(img):
    rgba = remove(img.convert("RGB"), session=SESSION)
    soft = np.array(rgba)[:, :, 3].astype(np.uint8)
    hard = soft > 128
    hard = ndimage.binary_fill_holes(hard)
    core = ndimage.binary_erosion(hard, iterations=2)
    M = np.where(core, 255, np.where(hard, soft, 0)).astype(np.uint8)
    ys, xs = np.where(hard)
    if len(xs) == 0: raise ValueError("No product detected in the photo.")
    box = (xs.min(), ys.min(), xs.max()+1, ys.max()+1)
    rgba = rgba.crop(box)
    Mm = Image.fromarray(M[box[1]:box[3], box[0]:box[2]])
    w, h = rgba.size; s = max(w, h)
    canvas = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    px, py = (s - w) // 2, s - h          
    canvas.paste(rgba, (px, py))
    Mc = Image.new("L", (s, s), 0)
    Mc.paste(Mm, (px, py))
    return canvas.resize((SIZE, SIZE), Image.LANCZOS), Mc.resize((SIZE, SIZE), Image.LANCZOS)

def detect_surface_y(bg_img, expected_bottom_y, x_center, x_width, search_range=140, edge_threshold=8.0):
    arr = np.array(bg_img.convert("L")).astype(np.float32)
    h, w = arr.shape
    half = max(x_width // 2, 150)
    c0, c1 = max(0, int(x_center - half)), min(w, int(x_center + half))
    grad = np.abs(arr[:-1, c0:c1] - arr[1:, c0:c1])
    grad = np.vstack([grad, np.zeros((1, c1 - c0))])
    row_strength = grad.mean(axis=1)
    smooth = ndimage.gaussian_filter1d(row_strength, sigma=2.5)
    start = max(0, expected_bottom_y - search_range)
    end = min(h, expected_bottom_y + search_range)
    window = smooth[start:end]
    if window.max() < edge_threshold:
        return expected_bottom_y
    peaks, props = find_peaks(window, distance=25, prominence=window.max() * 0.12)
    if len(peaks) == 0:
        return expected_bottom_y
    peak_ys = start + peaks
    distances = np.abs(peak_ys - expected_bottom_y)
    scores = props["prominences"] * np.exp(-distances / 80.0)
    best_y = int(peak_ys[np.argmax(scores)])
    return best_y if abs(best_y - expected_bottom_y) <= 180 else expected_bottom_y

def add_grounding(img, product, alpha, x, y, s):
    img = img.convert("RGBA")
    refl = product.transpose(Image.FLIP_TOP_BOTTOM)
    ra = np.array(alpha.transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float32) / 255.0
    ra *= np.linspace(0.22, 0.0, s)[:, None]
    center = s // 2
    half_w = s // 3
    h_fade = np.clip(1 - np.abs(np.arange(s) - center) / half_w, 0, 1)
    ra *= h_fade[None, :]
    img.paste(refl, (x, y + s - 4), Image.fromarray((ra * 255).astype(np.uint8)))
    band = max(12, s // 7)
    squashed = alpha.resize((s, band), Image.LANCZOS)
    sh = Image.new("L", (s, s), 0); sh.paste(squashed, (0, s - band))
    sh = sh.filter(ImageFilter.GaussianBlur(max(3, s // 30)))
    sh_arr = np.array(sh).astype(np.float32) / 255.0
    sh_center = s // 2
    sh_half = int(s // 1.8)
    sh_fade = np.clip(1 - np.abs(np.arange(s) - sh_center) / sh_half, 0.3, 1)
    sh_arr *= sh_fade[None, :]
    sh_arr = (sh_arr * 0.55 * 255).astype(np.uint8)
    black = Image.new("RGBA", (s, s), (10, 10, 10, 255))
    img.paste(black, (x, y + 2), Image.fromarray(sh_arr))
    return img

def make_font(font_file, size):
    try:
        if font_file is not None: return ImageFont.truetype(font_file.name, size)
        return ImageFont.truetype("arial.ttf", size)
    except Exception: return ImageFont.load_default()

def process_logo(logo_img, target_height):
    if logo_img is None: return None
    lg = logo_img.convert("RGBA")
    arr = np.array(lg)
    if arr[:, :, 3].mean() > 250:
        corners = [arr[0, 0, :3], arr[0, -1, :3], arr[-1, 0, :3], arr[-1, -1, :3]]
        bg = np.median(corners, axis=0)
        diff = np.linalg.norm(arr[:, :, :3].astype(float) - bg, axis=2)
        arr[diff < 35, 3] = 0
        lg = Image.fromarray(arr)
    ratio = target_height / lg.height
    lg = lg.resize((int(lg.width * ratio), target_height), Image.LANCZOS)
    return lg

def build_card(product, alpha, pal, logo, tagline, font_file):
    card = Image.new("RGBA", (W, H), ImageColor.getrgb(pal["bg"]))
    ps = int(SIZE * 0.9)
    p = product.resize((ps, ps), Image.LANCZOS); m = alpha.resize((ps, ps), Image.LANCZOS)
    card.paste(p, ((W-ps)//2, 130), m)
    d = ImageDraw.Draw(card)
    d.rectangle([0, 980, W, 994], fill=ImageColor.getrgb(pal["accent"]))
    if tagline:
        f = make_font(font_file, 52); tw = d.textlength(tagline, font=f)
        d.text(((W-tw)/2, 1020), tagline, font=f, fill=ImageColor.getrgb(pal["text"]))
    if logo is not None:
        lg = process_logo(logo, 110)
        if lg: card.paste(lg, ((W-lg.width)//2, 40), lg)
    return card.convert("RGB")

def _render(main, a2, a3, a4, a5, desc, scene, size_pct, height_pct, harmonize,
            use_brand, logo, c_primary, c_secondary, c_accent, c_bg, c_text,
            font_file, tagline, extra_assets, brand_card,
            use_custom, custom_pos, custom_neg, auto_place, contact_denoise):
    if main is None: raise ValueError("Upload a product photo first.")
    for i, ang in enumerate([a2, a3, a4, a5], 2):
        if ang is not None: ang.convert("RGB").save(f"identity_pack/angle_{i}.png")
    if font_file is not None:
        shutil.copy(font_file.name, os.path.join("brand_kit", "brand_font" + os.path.splitext(font_file.name)[1]))
    if extra_assets:
        for f in extra_assets: shutil.copy(f.name, os.path.join("brand_kit", os.path.basename(f.name)))

    product, alpha = prepare_product(main)
    s = int(SIZE * size_pct / 100)
    product = product.resize((s, s), Image.LANCZOS)
    alpha = alpha.resize((s, s), Image.LANCZOS)
    x = (W - s) // 2
    intended_y = int((H - s) * height_pct / 100)

    if use_custom and (custom_pos or "").strip():
        p1 = custom_pos.strip()
        p2 = p1 + ", product standing firmly on the surface, soft natural contact shadow, subtle reflection, photorealistic, high detail"
    else:
        p1t, p2t = SCENES[scene]
        p1 = p1t.format(**{k: random.choice(opts) for k, opts in VARIANTS[scene].items()})
        p2 = p2t.format(p=desc or "product")

    neg = (custom_neg or "").strip() if use_custom and (custom_neg or "").strip() else NEG
    if use_brand:
        p1 += f", elegant color palette of {cname(c_primary)}, {cname(c_secondary)} and {cname(c_accent)}, props and lighting matching these tones"

    bg = run(wf_background(p1, neg), "7").convert("RGBA")

    y = intended_y
    debug_bg = None
    if auto_place:
        intended_bottom = intended_y + s
        surface_y = detect_surface_y(bg, intended_bottom, x_center=x + s // 2, x_width=s)
        
        # --- DEBUG VISUALIZATION ---
        debug_bg = bg.copy()
        draw = ImageDraw.Draw(debug_bg)
        # Red line for intended bottom
        draw.line([(0, intended_bottom), (W, intended_bottom)], fill="red", width=4)
        # Green line for detected surface
        draw.line([(0, surface_y), (W, surface_y)], fill="lime", width=4)
        # Add small text labels
        try:
            font = ImageFont.load_default()
            draw.text((10, intended_bottom - 15), "Intended (Red)", fill="red", font=font)
            draw.text((10, surface_y - 15), "Detected (Green)", fill="lime", font=font)
        except Exception:
            pass
        # ---------------------------

        adjusted_y = surface_y - s
        if abs(adjusted_y - intended_y) >= 15:
            print(f"  📐 Auto-placement: y {intended_y} → {adjusted_y} (surface at {surface_y})")
            y = max(20, min(adjusted_y, H - s - 20))

    base = add_grounding(bg, product, alpha, x, y, s)
    base.paste(product, (x, y), alpha)

    if harmonize:
        an = upload("p_anchor.png", base)
        canvas_a = Image.new("L", (W, H), 0)
        canvas_a.paste(alpha, (x, y))
        ca = upload("p_alpha.png", canvas_a)
        seated = run(wf_seat(p2, an, ca, neg, contact_denoise), "12").convert("RGBA")
        base = seated.copy()
        base.paste(product, (x, y), alpha)

    final = base
    results = []
    
    # Prepend debug image if auto_place was used
    if debug_bg is not None:
        results.append(debug_bg.convert("RGB"))

    if use_brand:
        d = ImageDraw.Draw(final)
        if logo is not None:
            lg = process_logo(logo, int(H * 0.055))
            if lg: final.paste(lg, (W - lg.width - 40, H - lg.height - 40), lg)
        if tagline:
            f = make_font(font_file, 36); tw = d.textlength(tagline, font=f)
            d.text(((W - tw) / 2, H - 90), tagline, font=f, fill=c_text)
        results.append(final.convert("RGB"))
        if brand_card:
            results.append(build_card(product, alpha, {"bg": c_bg, "accent": c_accent, "text": c_text}, logo, tagline, font_file))
    else:
        results.append(final.convert("RGB"))
    return results

def generate(*args):
    try:
        return _render(*args), "Done ✅"
    except Exception as e:
        return [], f"Error: {e}  (Is ComfyUI Desktop running?)"

def generate_all(main, a2, a3, a4, a5, desc, scene_ignored, size_pct, height_pct, harmonize,
                 use_brand, logo, c_primary, c_secondary, c_accent, c_bg, c_text,
                 font_file, tagline, extra_assets, brand_card,
                 use_custom, custom_pos, custom_neg, auto_place, contact_denoise):
    try:
        out = []
        for sc in SCENES:
            print(f"🎬 Rendering scene: {sc}")
            out += _render(main, a2, a3, a4, a5, desc, sc, size_pct, height_pct, harmonize,
                           use_brand, logo, c_primary, c_secondary, c_accent, c_bg, c_text,
                           font_file, tagline, extra_assets, False,
                           False, "", "", auto_place, contact_denoise)
        return out, f"Asset pack done ✅ {len(out)} images"
    except Exception as e:
        return [], f"Error: {e}  (Is ComfyUI Desktop running?)"

with gr.Blocks(title="Mockup Factory v1.2") as ui:
    gr.Markdown("# 🏭 Mockup Factory — v1.2 (Bulletproof + Debug)")
    with gr.Row():
        with gr.Column():
            main = gr.Image(type="pil", label="Product photo (raw, any background)")
            with gr.Row():
                a2 = gr.Image(type="pil", label="Angle 2 (optional)")
                a3 = gr.Image(type="pil", label="Angle 3 (optional)")
            with gr.Row():
                a4 = gr.Image(type="pil", label="Angle 4 (optional)")
                a5 = gr.Image(type="pil", label="Angle 5 (optional)")
            desc   = gr.Textbox(label="Product description", value="lavender transparent pump bottle with a sage green pump")
            scene  = gr.Dropdown(list(SCENES), value="Kitchen", label="Scene")
            size_pct   = gr.Slider(50, 100, value=100, label="Product size")
            height_pct = gr.Slider(10, 70, value=45, label="Vertical position")
            auto_place = gr.Checkbox(label="Auto-detect surface (anti-float)", value=True)
            harmonize  = gr.Checkbox(label="AI harmonize pass (light-matching only)", value=False)
            contact_denoise = gr.Slider(0.10, 0.40, value=0.25, step=0.05, label="Contact denoise (harmonize strength)")
            with gr.Accordion("Brand kit", open=False):
                use_brand = gr.Checkbox(label="Add brand layer")
                logo = gr.Image(type="pil", label="Logo (transparent PNG)", image_mode="RGBA")
                with gr.Row():
                    c_primary   = gr.ColorPicker(value="#8A6E95", label="Primary")
                    c_secondary = gr.ColorPicker(value="#B2BEB5", label="Secondary")
                    c_accent    = gr.ColorPicker(value="#5B7C3C", label="Accent")
                with gr.Row():
                    c_bg   = gr.ColorPicker(value="#F5F5F5", label="Card background")
                    c_text = gr.ColorPicker(value="#D9D2C6", label="Text color")
                font_file = gr.File(label="Brand font (.ttf / .otf)", file_types=[".ttf", ".otf"])
                tagline = gr.Textbox(label="Tagline", value="")
                extra_assets = gr.File(label="Other brand assets", file_count="multiple")
                brand_card = gr.Checkbox(label="Also generate flat brand-color ad card", value=True)
            with gr.Accordion("Advanced: custom prompts (Pro Mode)", open=False):
                use_custom = gr.Checkbox(label="Use my own prompts instead of the scene preset")
                custom_pos = gr.Textbox(label="Positive prompt", lines=3)
                custom_neg = gr.Textbox(label="Negative prompt", lines=2)
            btn = gr.Button("GENERATE 🚀", variant="primary")
            btn_all = gr.Button("GENERATE ALL SCENES 🎬  (asset pack)", variant="secondary")
        with gr.Column():
            out_img = gr.Gallery(label="Results", columns=1, object_fit="contain", height=900)
            status  = gr.Textbox(label="Status")
    INPUTS = [main, a2, a3, a4, a5, desc, scene, size_pct, height_pct, harmonize,
              use_brand, logo, c_primary, c_secondary, c_accent, c_bg, c_text,
              font_file, tagline, extra_assets, brand_card,
              use_custom, custom_pos, custom_neg, auto_place, contact_denoise]
    btn.click(generate, INPUTS, [out_img, status])
    btn_all.click(generate_all, INPUTS, [out_img, status])

ui.launch(inbrowser=False)