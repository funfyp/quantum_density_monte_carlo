"""
Simple FastAPI backend for the web demo prototype.

Exposes POST /run which expects a JSON body with simulation parameters.
It calls src.full_monte.run_grid_sim(params) if available and produces PNG
heatmaps for entropy, expected points, and composite F. The images are
returned base64-encoded along with simple summary statistics.

This file is meant as a lightweight prototype — adapt error handling,
parameter validation and security (CORS, auth, input sanitization) for
production use.
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import base64
import io
import traceback
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Any, Dict

app = FastAPI(title="Quantum density Monte Carlo demo")

# Allow all origins for the prototype. Lock this down in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def fig_to_png_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return b64


def heatmap_image(arr: np.ndarray, title: str = "", cmap: str = "viridis") -> str:
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    im = ax.imshow(arr, origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig_to_png_base64(fig)


def points_image(grid_shape, points, title: str = "Expected points") -> str:
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.set_xlim(0, grid_shape[1])
    ax.set_ylim(0, grid_shape[0])
    ax.set_title(title)
    ax.invert_yaxis()
    if points is not None and len(points) > 0:
        pts = np.array(points)
        if pts.ndim == 1 and pts.size == 2:
            pts = pts.reshape((1, 2))
        ax.scatter(pts[:, 0], pts[:, 1], c="red", s=10)
    ax.axis("off")
    return fig_to_png_base64(fig)


@app.post("/run")
async def run(request: Request) -> Dict[str, Any]:
    """Run a simulation and return base64-encoded PNGs and summary stats.

    Body: arbitrary JSON parameters forwarded to src.full_monte.run_grid_sim.
    Response: {
        "images": {"entropy": <base64>, "expected": <base64>, "composite": <base64>},
        "summary": { ... }
    }
    """
    params = await request.json()

    # Try to call the project's Python simulation function. Provide a simple
    # fallback if the module isn't importable so the demo still works.
    try:
        from src.full_monte import run_grid_sim  # type: ignore
    except Exception as e:  # pragma: no cover - runtime environment dependent
        # Fallback mock: produce synthetic arrays based on params
        def run_grid_sim(p):
            size = int(p.get("grid_size", 128))
            # entropy-like field: gaussian + noise
            x = np.linspace(-3, 3, size)
            X, Y = np.meshgrid(x, x)
            entropy = np.exp(-(X ** 2 + Y ** 2))
            entropy += 0.1 * np.random.RandomState(0).randn(size, size)
            # expected points: a few sample points in grid coordinates
            expected = np.array([[size * 0.3, size * 0.6], [size * 0.7, size * 0.4]])
            composite = entropy * (1 + 0.3 * np.sin(3 * X) * np.cos(3 * Y))
            return {
                "entropy": entropy,
                "expected": expected,
                "composite": composite,
                "meta": {"note": "mocked run (src.full_monte not importable)"},
            }

        run_grid_sim = run_grid_sim

    try:
        result = run_grid_sim(params)
    except Exception:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail={"error": "simulation failed", "trace": tb})

    # Expecting result to contain numpy arrays for entropy and composite, and pts
    entropy = np.asarray(result.get("entropy"))
    composite = np.asarray(result.get("composite"))
    expected = result.get("expected")

    # Build images
    try:
        entropy_img = heatmap_image(entropy, title="Entropy")
        composite_img = heatmap_image(composite, title="Composite F")
        expected_img = points_image(entropy.shape, expected, title="Expected points")
    except Exception:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail={"error": "image generation failed", "trace": tb})

    # summary stats (simple)
    summary = {
        "entropy_mean": float(np.nanmean(entropy)),
        "entropy_min": float(np.nanmin(entropy)),
        "entropy_max": float(np.nanmax(entropy)),
        "expected_count": int(np.shape(expected)[0]) if expected is not None else 0,
    }
    # merge any meta from result
    if isinstance(result.get("meta"), dict):
        summary.update(result.get("meta"))

    return {
        "images": {
            "entropy": entropy_img,
            "expected": expected_img,
            "composite": composite_img,
        },
        "summary": summary,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("viz.web.backend:app", host="0.0.0.0", port=8000, reload=True)
