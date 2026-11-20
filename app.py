"""Interactive seam carving app (bonus Item 7b). Owner: P6.

Run with:
    streamlit run app.py

Upload an image (or use the sample), pick a seam finder and the target size,
then see the result, every removed seam, and a step-by-step animation.
The "Compare algorithms" tab runs every finished finder on the same image.
"""

import base64
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from seam_carving.core import carve
from seam_carving.finders import FINDERS, get_finder
from seam_carving.seam_utils import seam_cost
from seam_carving.visualize import SeamRecorder, all_seams_on_original, make_gif, pad_to

SAMPLE = Path(__file__).parent / "images" / "sample.png"

# Graph cut builds a graph with one node per pixel, so it is far slower.
# Above this many pixels we warn the user (and suggest a smaller image).
GRAPHCUT_PIXEL_LIMIT = 60 * 60


# ---------- plain functions (no Streamlit), tested in tests/test_app.py ----------

def decode_image(data):
    """Bytes of an uploaded PNG/JPG -> H x W x 3 uint8 RGB array."""
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read the uploaded file as an image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def encode_png(img):
    """RGB array -> PNG bytes (for the download buttons)."""
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError("Could not encode image as PNG")
    return buf.tobytes()


def limit_size(img, max_side):
    """Shrink img (keeping its aspect ratio) so its longest side is <= max_side.

    Our carve loop recomputes the energy and runs the finder once per seam,
    so a big photo can take a long time. INTER_AREA is the usual choice for
    shrinking without jagged edges.
    """
    height, width = img.shape[:2]
    if max(height, width) <= max_side:
        return img
    factor = max_side / max(height, width)
    size = (max(2, round(width * factor)), max(2, round(height * factor)))  # cv2 wants (w, h)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def run_carve(img, algo, new_width, new_height, keep_frames=True):
    """Carve img with one algorithm and collect everything the app shows.

    Returns a dict with the result, the all-seams picture, the step frames,
    the run time in seconds and the total energy of all removed seams
    (lower = the algorithm removed less important pixels).
    """
    finder = get_finder(algo)
    costs = []

    def finder_with_cost(energy):
        # Wrap the real finder so we can record each seam's energy (what DP
        # minimises) without changing any of the finder code.
        seam = finder(energy)
        costs.append(seam_cost(energy, np.asarray(seam)))
        return seam

    recorder = SeamRecorder(keep_frames=keep_frames)
    start = time.perf_counter()
    result = carve(img, new_width, new_height, finder_with_cost, on_seam=recorder)
    elapsed = time.perf_counter() - start

    return {
        "result": result,
        "all_seams": all_seams_on_original(img, recorder.removed),
        "frames": recorder.frames,
        "seconds": elapsed,
        "seams": len(recorder.removed),
        "total_energy": float(sum(costs)),
    }


def gif_bytes(frames, final_frame, every, scale):
    """make_gif() writes to a file, so write to a temp file and read it back."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "steps.gif"
        make_gif(frames, path, every=every, scale=scale, final_frame=final_frame)
        return path.read_bytes()


def display_scale(img, min_width=360):
    """Small images look tiny in the browser; enlarge them by a whole number
    (nearest neighbour, so 1-pixel seams stay sharp)."""
    return max(1, min_width // img.shape[1])


def enlarge(img, scale):
    return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1) if scale > 1 else img


# ---------- Streamlit user interface ----------

@st.cache_data(show_spinner=False)
def cached_run(img, algo, new_width, new_height, keep_frames):
    # Streamlit reruns the whole script after every click, so cache results:
    # the same image + settings is only carved once.
    return run_carve(img, algo, new_width, new_height, keep_frames)


def available_finders():
    """Finders that are actually written (unfinished ones raise NotImplementedError)."""
    names = []
    tiny = np.random.default_rng(0).random((3, 3))
    for name in FINDERS:
        try:
            get_finder(name)(tiny)
            names.append(name)
        except NotImplementedError:
            pass
    return names


@st.cache_data(show_spinner=False)
def cached_gif(frames, final_frame, every, scale):
    # Without this the GIF is rebuilt on every slider move (Streamlit reruns the script).
    return gif_bytes(frames, final_frame, every, scale)


def show_image(img, caption, scale):
    # Use ONE scale (from the original image) for every picture, so a narrow
    # result is not blown up more than the original it came from.
    big = enlarge(img, scale)
    # Give the exact pixel width, otherwise Streamlit may stretch narrow images.
    st.image(big, caption=caption, width=big.shape[1])


def main():
    st.set_page_config(page_title="Seam Carving", layout="wide")
    st.title("Seam carving: content-aware image resizing")

    # ----- input image -----
    st.sidebar.header("1. Image")
    upload = st.sidebar.file_uploader("Upload a PNG or JPG", type=["png", "jpg", "jpeg"])
    try:
        img = decode_image(upload.getvalue()) if upload else decode_image(SAMPLE.read_bytes())
    except ValueError as err:
        st.error(str(err))
        return
    if upload is None:
        st.sidebar.caption("Using images/sample.png")

    max_side = st.sidebar.slider("Shrink input so longest side is at most", 50, 800, 300, step=10,
                                 help="Carving is slow on big photos, so large uploads are scaled down first.")
    img = limit_size(img, max_side)
    height, width = img.shape[:2]
    st.sidebar.caption(f"Working size: {width} x {height}")
    scale = display_scale(img)
    if width < 3 or height < 3:
        st.error("The image is too small to carve. Use a bigger image or raise the size slider.")
        return

    # ----- target size -----
    st.sidebar.header("2. Target size")
    new_width = st.sidebar.slider("Width", 2, width, max(2, int(width * 0.75)))
    new_height = st.sidebar.slider("Height", 2, height, height)
    st.sidebar.caption(f"Removes {width - new_width} vertical and {height - new_height} horizontal seams")

    finders = available_finders()

    tab_resize, tab_compare = st.tabs(["Resize", "Compare algorithms"])

    # ----- tab 1: one algorithm, full visualisation -----
    with tab_resize:
        algo = st.selectbox("Seam finder", finders, index=finders.index("dp") if "dp" in finders else 0)
        if algo == "graphcut" and width * height > GRAPHCUT_PIXEL_LIMIT:
            st.warning(f"Graph cut is slow on a {width} x {height} image. "
                       "Try lowering the 'longest side' slider first.")

        if new_width == width and new_height == height:
            st.info("Move the Width or Height slider to choose a smaller size.")
        else:
            with st.spinner(f"Carving with {algo}..."):
                run = cached_run(img, algo, new_width, new_height, True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Seams removed", run["seams"])
            c2.metric("Time", f"{run['seconds']:.2f} s")
            c3.metric("Energy removed", f"{run['total_energy']:.0f}",
                      help="Sum of the energy of every removed seam. Lower = less important pixels removed.")

            left, mid, right = st.columns(3)
            with left:
                show_image(img, f"Original {width} x {height}", scale)
            with mid:
                show_image(run["all_seams"], "All removed seams (red = vertical, blue = horizontal)", scale)
            with right:
                show_image(run["result"], f"Result {new_width} x {new_height}", scale)
                st.download_button("Download result", encode_png(run["result"]),
                                   file_name=f"carved_{algo}_{new_width}x{new_height}.png", mime="image/png")

            st.subheader("Step by step")
            frames = run["frames"]
            # A slider needs min < max, so only show it when there is more than one step.
            step = st.slider("Seam", 1, len(frames), 1) if len(frames) > 1 else 1
            # Pad every step back to the original size (like the GIF), so the picture
            # stays the same size while you scrub and you can see it shrinking.
            show_image(pad_to(frames[step - 1], height, width, fill=220),  # light grey, visible on the white page
                       f"Step {step} of {len(frames)}: this seam is removed next", scale)

            if st.checkbox("Make an animated GIF"):
                every = 1
                if len(frames) >= 4:
                    every = st.slider("Keep every Nth step", 1, len(frames) // 2, max(1, len(frames) // 40))
                data = cached_gif(frames, run["result"], every, scale)
                # st.image can drop the animation, so embed the GIF directly.
                b64 = base64.b64encode(data).decode()
                st.markdown(f'<img src="data:image/gif;base64,{b64}">', unsafe_allow_html=True)
                st.download_button("Download GIF", data, file_name=f"steps_{algo}.gif", mime="image/gif")

    # ----- tab 2: compare every finished algorithm on the same input -----
    with tab_compare:
        st.write("Runs every finished seam finder with the same image and target size.")
        chosen = st.multiselect("Algorithms", finders, default=[f for f in finders if f != "graphcut"])
        if "graphcut" in chosen and width * height > GRAPHCUT_PIXEL_LIMIT:
            st.warning("Graph cut is included and the image is large, so this may take a while.")

        if st.button("Run comparison", disabled=not chosen or (new_width == width and new_height == height)):
            runs = {}
            for name in chosen:
                with st.spinner(f"Carving with {name}..."):
                    runs[name] = cached_run(img, name, new_width, new_height, False)

            st.table({
                "Algorithm": list(runs),
                "Time (s)": [f"{r['seconds']:.3f}" for r in runs.values()],
                "Energy removed": [f"{r['total_energy']:.0f}" for r in runs.values()],
            })
            for column, (name, r) in zip(st.columns(len(runs)), runs.items()):
                with column:
                    show_image(r["all_seams"], f"{name}: seams", scale)
                    show_image(r["result"], f"{name}: result", scale)


if __name__ == "__main__":
    main()
