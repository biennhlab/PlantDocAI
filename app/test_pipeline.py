"""Quick test of full app pipeline."""
import sys; sys.path.insert(0, ".")
from src.evaluation.predictor import InferencePipeline
from PIL import Image

# Test pipeline load
p = InferencePipeline("artifacts/efficientnet_extended_artifacts", device="cpu")
print("Model:", p.modelName, "| Classes:", p.numClasses)

# Test prediction
img = Image.open(r"imageTest\3_Tomato___Bacterial_spot\Tomato___Bacterial_spot.jpg").convert("RGB")
result = p.predictFromPil(img, topK=5)
preds = result["predictions"]
for x in preds:
    print(f"  {x['className']:50s} {x['confidence']*100:6.2f}%")
print(f"  OOD: is_ood={result['ood']['is_ood']}, energy={result['ood']['energy_score']:.4f}")

# Test Grad-CAM
explain_result = p.explainFromPil(img, topK=3)
print("GradCAM overlay size:", explain_result["gradcamOverlay"].size)

# Test recommendation import
from app.recommendations import getRecommendation
rec = getRecommendation(preds[0]["className"])
print("Has recommendation:", rec is not None)
if rec:
    print("Disease name:", rec.get("name", "N/A"))
    print("Has symptoms:", "symptoms" in rec)
    print("Has treatment:", "treatment" in rec)

print("\nALL TESTS PASSED")
