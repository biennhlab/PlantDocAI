"""Quick smoke test for the inference pipeline used by app.py."""
import sys
sys.path.insert(0, ".")

from src.evaluation.predictor import InferencePipeline
from PIL import Image

print("[1] Loading pipeline...")
p = InferencePipeline("artifacts/mobilenetV2_extended_artifacts", device="cpu")
print(f"    Model: {p.modelName}, Classes: {p.numClasses}")

print("[2] Testing predict...")
result = p.predict("imageTest/Tomato bacterial_spot.png", topK=3)
for x in result["predictions"]:
    print(f"    {x['className']}: {x['confidence']*100:.1f}%")
print(f"    OOD: is_ood={result['ood']['is_ood']}, energy={result['ood']['energy_score']:.4f}")

print("[3] Testing explainFromPil...")
img = Image.open("imageTest/Tomato bacterial_spot.png").convert("RGB")
result = p.explainFromPil(img, topK=3)
print(f"    Predictions: {len(result['predictions'])}")
print(f"    Overlay type: {type(result['gradcamOverlay'])}")
print(f"    Overlay size: {result['gradcamOverlay'].size}")
print(f"    Target class idx: {result['targetClassIdx']}")

print("[4] Testing RGBA image...")
rgba = Image.open("imageTest/pic1.png")
print(f"    Mode: {rgba.mode}")
result = p.predictFromPil(rgba, topK=1)
top1 = result["predictions"][0]
print(f"    Result: {top1['className']}: {top1['confidence']*100:.1f}%")

print("\n[OK] All tests passed.")
