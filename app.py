# app.py
"""
PlantDocAI — Ứng dụng nhận diện bệnh lá cây tích hợp AI giải thích được.

Tái sử dụng hoàn toàn InferencePipeline từ src/evaluation/predictor.py.
Không viết lại logic predict — chỉ gọi pipeline API.

Chạy: streamlit run app.py
"""

import sys
import math
import logging
from pathlib import Path

# Đảm bảo project root trong sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from PIL import Image

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "tiff"}
MIN_IMAGE_SIZE = 32
MAX_FILE_SIZE_MB = 10

# Confidence thresholds
HIGH_CONFIDENCE = 0.80
LOW_CONFIDENCE = 0.50
VERY_LOW_CONFIDENCE = 0.30

# Entropy threshold cho OOD warning (uniform distribution trên 38 classes ~ 5.25)
ENTROPY_WARNING_RATIO = 0.55  # cảnh báo nếu entropy > 55% của max entropy


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

APP_CSS = """
<style>
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.85rem !important;
        color: #6c757d !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #212529 !important;
    }

    /* Section dividers */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        padding: 8px 0 4px 0;
        margin-top: 12px;
        border-bottom: 2px solid #e9ecef;
        margin-bottom: 12px;
    }

    /* Recommendation cards */
    .reco-card {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 8px 0;
    }
    .reco-card.warning {
        border-left-color: #ffc107;
    }
    .reco-card.danger {
        border-left-color: #dc3545;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        color: #adb5bd;
        font-size: 0.8rem;
        padding: 20px 0 10px 0;
        border-top: 1px solid #e9ecef;
        margin-top: 30px;
    }

    /* Hide default streamlit hamburger for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _formatClassName(rawName: str) -> str:
    """Chuyển 'Plant___Condition' thành 'Plant — Condition' dễ đọc."""
    parts = rawName.split("___")
    if len(parts) == 2:
        plant = parts[0].replace("_", " ")
        condition = parts[1].replace("_", " ").strip()
        return f"{plant} — {condition}"
    return rawName.replace("_", " ")


def _computeEntropy(predictions: list) -> float:
    """Tính entropy chuẩn hóa từ predictions. 0 = chắc chắn, 1 = uniform."""
    if not predictions:
        return 1.0
    n = len(predictions)
    if n <= 1:
        return 0.0
    entropy = 0.0
    for p in predictions:
        conf = p["confidence"]
        if conf > 1e-10:
            entropy -= conf * math.log(conf)
    maxEntropy = math.log(n)
    return entropy / maxEntropy if maxEntropy > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _loadPipeline(modelDir: str):
    """Load InferencePipeline — cached, chỉ chạy 1 lần per model dir."""
    from src.evaluation.predictor import InferencePipeline
    return InferencePipeline(modelDir=modelDir, device="cpu")


def _discoverArtifactDirs() -> list:
    """Tìm tất cả artifact dirs có config.json + checkpoint."""
    dirs = []
    if not ARTIFACTS_DIR.exists():
        return dirs
    for d in sorted(ARTIFACTS_DIR.iterdir()):
        if not d.is_dir():
            continue
        hasConfig = (d / "config.json").exists()
        hasCheckpoint = any([
            (d / "checkpoints" / "best.pt").exists(),
            (d / "best.pt").exists(),
            (d / "checkpoints" / "last.pt").exists(),
        ])
        if hasConfig and hasCheckpoint:
            dirs.append(d.name)
    return dirs


# ─────────────────────────────────────────────────────────────────────────────
# Image Validation
# ─────────────────────────────────────────────────────────────────────────────

def _validateImage(uploadedFile) -> tuple:
    """Validate ảnh upload. Returns (image, warnings: list, error: str|None)."""
    warnings = []

    if uploadedFile is None:
        return None, [], "Chưa có ảnh nào được tải lên."

    filename = uploadedFile.name.lower()
    ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return None, [], (
            f"Định dạng file không được hỗ trợ (.{ext}). "
            f"Vui lòng dùng: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    fileSize = uploadedFile.size
    if fileSize > MAX_FILE_SIZE_MB * 1024 * 1024:
        return None, [], (
            f"File quá lớn ({fileSize / 1024 / 1024:.1f} MB). "
            f"Giới hạn: {MAX_FILE_SIZE_MB} MB."
        )

    try:
        image = Image.open(uploadedFile)
        image.load()
    except Exception:
        return None, [], "Không thể đọc file ảnh. File có thể bị hỏng."

    w, h = image.size
    if w < MIN_IMAGE_SIZE or h < MIN_IMAGE_SIZE:
        return None, [], (
            f"Ảnh quá nhỏ ({w}×{h} px). "
            f"Kích thước tối thiểu: {MIN_IMAGE_SIZE}×{MIN_IMAGE_SIZE} px."
        )

    # Quality warnings (non-blocking)
    if w < 100 or h < 100:
        warnings.append("Ảnh có độ phân giải thấp — kết quả có thể kém chính xác.")
    if max(w, h) / min(w, h) > 3:
        warnings.append("Ảnh có tỉ lệ bất thường — nên dùng ảnh vuông hoặc gần vuông.")

    try:
        image = image.convert("RGB")
    except Exception:
        return None, [], "Không thể chuyển đổi ảnh sang RGB."

    return image, warnings, None


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def _runAnalysis(pipeline, image, topK, showGradCAM, gradcamAlpha):
    """Chạy inference. Returns (predictions, gradcamOverlay, error)."""
    gradcamOverlay = None
    predictions = None
    gradcamError = False

    try:
        if showGradCAM:
            result = pipeline.explainFromPil(image, topK=topK, alpha=gradcamAlpha)
            predictions = result["predictions"]
            gradcamOverlay = result["gradcamOverlay"]
        else:
            predictions = pipeline.predictFromPil(image, topK=topK)
    except Exception:
        if showGradCAM:
            gradcamError = True
            try:
                predictions = pipeline.predictFromPil(image, topK=topK)
            except Exception as e2:
                return None, None, str(e2)
        else:
            import traceback
            return None, None, traceback.format_exc()

    return predictions, gradcamOverlay, "gradcam_fallback" if gradcamError else None


# ─────────────────────────────────────────────────────────────────────────────
# Render Functions
# ─────────────────────────────────────────────────────────────────────────────

def _renderSidebar(artifactDirs):
    """Sidebar: model selector, config, info."""
    with st.sidebar:
        st.markdown("## 🌿 PlantDoc AI")
        st.caption("Hệ thống nhận diện bệnh lá cây")
        st.divider()

        # Model selector
        st.markdown("### ⚙️ Cấu hình")
        defaultIdx = 0
        for i, name in enumerate(artifactDirs):
            if "extended" in name.lower():
                defaultIdx = i
                break

        selectedDir = st.selectbox(
            "Mô hình", artifactDirs, index=defaultIdx,
            help="Chọn artifact directory chứa model checkpoint.",
        )
        topK = st.slider("Top-K dự đoán", 1, 10, 5)
        showGradCAM = st.checkbox(
            "Hiển thị Grad-CAM", value=True,
            help="Trực quan hóa vùng ảnh mô hình tập trung khi dự đoán.",
        )
        gradcamAlpha = st.slider(
            "Độ đậm Grad-CAM", 0.2, 0.8, 0.5, 0.05,
            disabled=not showGradCAM,
        )

        st.divider()

        # Model info — sẽ được cập nhật sau khi load pipeline
        st.markdown("### 📋 Thông tin mô hình")
        modelInfoPlaceholder = st.empty()

        st.divider()
        st.caption(
            "⚠️ **Lưu ý:** Đây là công cụ hỗ trợ nghiên cứu, "
            "không thay thế ý kiến chuyên gia nông nghiệp."
        )

    return selectedDir, topK, showGradCAM, gradcamAlpha, modelInfoPlaceholder


def _renderModelInfo(placeholder, pipeline):
    """Hiển thị model info trong sidebar placeholder."""
    import torch
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    placeholder.markdown(f"""
- **Kiến trúc:** `{pipeline.modelName}`
- **Số lớp:** {pipeline.numClasses}
- **Input size:** {pipeline.imageSize}×{pipeline.imageSize}
- **Device:** {device}
""")


def _renderHeader():
    """Header + giới thiệu."""
    st.title("🌿 PlantDoc AI")
    st.markdown(
        "**Hệ thống nhận diện bệnh lá cây bằng Deep Learning** "
        "tích hợp **Explainable AI** (Grad-CAM) và khuyến nghị xử lý."
    )
    with st.expander("📖 Hướng dẫn sử dụng", expanded=False):
        st.markdown("""
1. **Chọn mô hình** ở thanh bên trái (sidebar).
2. **Tải ảnh lá cây** lên — hệ thống sẽ tự động phân tích.
3. Xem **kết quả dự đoán**, **Grad-CAM**, và **khuyến nghị**.

**Lưu ý quan trọng:**
- Nên dùng ảnh chụp cận cảnh lá cây, đủ sáng và rõ nét.
- Kết quả chỉ mang tính tham khảo — luôn tham vấn chuyên gia nông nghiệp.
- Mô hình chỉ nhận diện các bệnh trong tập dữ liệu đã train (PlantVillage).
""")


def _renderPredictionResults(predictions):
    """Hiển thị kết quả dự đoán: metric cards + top-K."""
    top1 = predictions[0]
    confidence = top1["confidence"]
    confPct = confidence * 100
    displayName = _formatClassName(top1["className"])

    # Confidence color/label
    if confidence >= HIGH_CONFIDENCE:
        confLabel = "Cao"
        confDelta = "✓ Đáng tin cậy"
    elif confidence >= LOW_CONFIDENCE:
        confLabel = "Trung bình"
        confDelta = "~ Nên kiểm tra thêm"
    else:
        confLabel = "Thấp"
        confDelta = "⚠ Cần tham khảo chuyên gia"

    # Metric cards
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🏷️ Dự đoán chính", displayName)
    with col2:
        st.metric("📊 Độ tin cậy", f"{confPct:.1f}%", delta=confDelta)

    # Top-K results
    if len(predictions) > 1:
        st.markdown('<p class="section-header">📋 Kết quả Top-K</p>',
                    unsafe_allow_html=True)
        for i, pred in enumerate(predictions):
            conf = pred["confidence"] * 100
            name = _formatClassName(pred["className"])
            medals = {0: "🥇", 1: "🥈", 2: "🥉"}
            prefix = medals.get(i, f"&nbsp;{i+1}.")
            st.progress(
                pred["confidence"],
                text=f"{prefix} {name} — {conf:.1f}%",
            )


def _renderGradcamSection(image, gradcamOverlay, gradcamFallback, showGradCAM):
    """Hiển thị Grad-CAM section."""
    st.markdown('<p class="section-header">🔬 Giải thích mô hình (Grad-CAM)</p>',
                unsafe_allow_html=True)

    if gradcamOverlay is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Ảnh gốc", use_container_width=True)
        with col2:
            st.image(gradcamOverlay,
                     caption="Grad-CAM — Vùng mô hình tập trung",
                     use_container_width=True)
        st.info(
            "**Grad-CAM** cho thấy vùng ảnh mô hình \"chú ý\" khi đưa ra dự đoán. "
            "Vùng đỏ/vàng = mô hình tập trung cao, vùng xanh = tập trung thấp.\n\n"
            "*Đây là trực quan hóa hậu nghiệm (post-hoc), không phải bằng chứng "
            "nhân quả. Kết quả chỉ mang tính hỗ trợ giải thích.*",
            icon="ℹ️",
        )
    elif gradcamFallback:
        st.warning(
            "Grad-CAM gặp lỗi kỹ thuật trong lần phân tích này. "
            "Kết quả dự đoán vẫn hiển thị bình thường.",
            icon="⚠️",
        )
    elif showGradCAM:
        st.info("Grad-CAM không khả dụng cho lần phân tích này.", icon="ℹ️")


def _renderRecommendation(className):
    """Hiển thị khuyến nghị có cấu trúc."""
    from app.recommendations import getRecommendation

    rec = getRecommendation(className)
    if rec is None:
        st.info(
            "Chưa có thông tin khuyến nghị cho lớp này. "
            "Vui lòng tham khảo chuyên gia nông nghiệp.",
            icon="ℹ️",
        )
        return

    st.markdown('<p class="section-header">💡 Khuyến nghị</p>',
                unsafe_allow_html=True)

    isHealthy = rec.get("isHealthy", False)

    if isHealthy:
        st.success(f"✅ **{rec['name']}** — Cây khỏe mạnh!", icon="🌱")
        st.markdown(f"**Tiếp tục chăm sóc:** {rec.get('treatment', '')}")
        if rec.get("prevention"):
            st.markdown(f"**Phòng ngừa:** {rec['prevention']}")
    else:
        st.warning(f"🔍 **{rec['name']}**", icon="🔬")

        # Structured display
        tabSymp, tabTreat, tabPrev = st.tabs([
            "🔎 Triệu chứng", "💊 Gợi ý xử lý", "🛡️ Phòng ngừa"
        ])
        with tabSymp:
            st.markdown(rec.get("symptoms", "Chưa có thông tin."))
        with tabTreat:
            st.markdown(rec.get("treatment", "Tham khảo chuyên gia nông nghiệp."))
        with tabPrev:
            st.markdown(rec.get("prevention", "Tham khảo chuyên gia nông nghiệp."))

        if rec.get("notes"):
            st.caption(f"📝 *{rec['notes']}*")

    # Disclaimer
    st.caption(
        "*Khuyến nghị mang tính tham khảo. Luôn tham vấn chuyên gia "
        "nông nghiệp trước khi áp dụng biện pháp xử lý.*"
    )


def _renderWarnings(predictions, imageWarnings):
    """Hiển thị cảnh báo confidence và chất lượng ảnh."""
    hasWarning = False
    top1 = predictions[0]
    confidence = top1["confidence"]

    # Image quality warnings
    for w in imageWarnings:
        st.warning(f"📷 {w}", icon="⚠️")
        hasWarning = True

    # Confidence warnings
    if confidence < VERY_LOW_CONFIDENCE:
        st.error(
            "**Độ tin cậy rất thấp** — Mô hình không tự tin với dự đoán này. "
            "Ảnh có thể không phải lá cây, không rõ nét, hoặc nằm ngoài "
            "phân phối dữ liệu đã train. Kết quả chỉ nên dùng để tham khảo.",
            icon="🚨",
        )
        hasWarning = True
    elif confidence < LOW_CONFIDENCE:
        st.warning(
            "**Độ tin cậy thấp** — Kết quả có thể không chính xác. "
            "Hãy thử chụp ảnh rõ hơn hoặc tham khảo chuyên gia.",
            icon="⚠️",
        )
        hasWarning = True

    # Entropy-based OOD warning
    normalizedEntropy = _computeEntropy(predictions)
    if normalizedEntropy > ENTROPY_WARNING_RATIO and confidence < HIGH_CONFIDENCE:
        st.warning(
            "**Phân bố dự đoán phân tán** — Mô hình không thể phân biệt rõ ràng "
            "giữa các lớp. Ảnh có thể không phải lá cây hoặc không thuộc "
            "các loại bệnh đã train.",
            icon="🔀",
        )
        hasWarning = True

    return hasWarning


def _renderFooter(pipeline):
    """Footer với thông tin model và giới hạn."""
    st.divider()
    with st.expander("ℹ️ Thông tin mô hình & Giới hạn"):
        st.markdown(f"""
**Thông tin mô hình:**
- **Kiến trúc:** {pipeline.modelName} (timm)
- **Tập dữ liệu:** PlantVillage Extended ({pipeline.numClasses} lớp)
- **Tiền xử lý:** Resize → CenterCrop({pipeline.imageSize}) → Normalize(ImageNet)

**Giới hạn hiện tại:**
- Chỉ nhận diện {pipeline.numClasses} loại bệnh/trạng thái đã được train
- Ảnh ngoài phân phối dữ liệu train có thể cho kết quả sai
- Không thay thế chẩn đoán của chuyên gia nông nghiệp
- Grad-CAM là trực quan hóa hậu nghiệm, không phải bằng chứng nhân quả

**Dự án:** PlantDoc AI — Thực tập cơ sở (INT13147)
""")

    st.markdown(
        '<div class="app-footer">'
        'PlantDoc AI © 2026 — Hệ thống hỗ trợ nghiên cứu, không thay thế chuyên gia.'
        '</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="PlantDoc AI — Nhận diện bệnh lá cây",
        page_icon="🌿",
        layout="wide",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)

    # ── Discover models ──────────────────────────────────────────────────
    artifactDirs = _discoverArtifactDirs()
    if not artifactDirs:
        st.error(
            "❌ Không tìm thấy artifact nào trong thư mục `artifacts/`. "
            "Hãy đảm bảo có ít nhất một thư mục chứa `config.json` và checkpoint."
        )
        st.stop()

    # ── Sidebar ──────────────────────────────────────────────────────────
    selectedDir, topK, showGradCAM, gradcamAlpha, modelInfoPlaceholder = \
        _renderSidebar(artifactDirs)

    # ── Header ───────────────────────────────────────────────────────────
    _renderHeader()

    # ── Load model ───────────────────────────────────────────────────────
    modelDirPath = str(ARTIFACTS_DIR / selectedDir)
    try:
        with st.spinner("🔄 Đang tải mô hình..."):
            pipeline = _loadPipeline(modelDirPath)
        _renderModelInfo(modelInfoPlaceholder, pipeline)
    except FileNotFoundError as e:
        st.error(f"❌ Không tìm thấy file cần thiết: {e}")
        st.info("Kiểm tra lại thư mục artifacts/ và đảm bảo có config.json + checkpoint.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Lỗi khi tải mô hình: {e}")
        logger.exception("Failed to load pipeline")
        st.stop()

    # ── Upload ───────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📤 Tải ảnh lên</p>',
                unsafe_allow_html=True)
    uploadedFile = st.file_uploader(
        "Chọn ảnh lá cây cần phân tích",
        type=list(ALLOWED_EXTENSIONS),
        help="Hỗ trợ JPG, PNG, WebP, BMP, TIFF. Tối đa 10 MB.",
        label_visibility="collapsed",
    )

    if uploadedFile is None:
        st.session_state.pop("results", None)
        st.info(
            "👆 **Tải ảnh lá cây lên để bắt đầu phân tích.**\n\n"
            "Nên dùng ảnh cận cảnh lá cây, đủ sáng và rõ vùng bệnh (nếu có).",
            icon="🌿",
        )
        st.stop()

    # ── Validate ─────────────────────────────────────────────────────────
    image, imageWarnings, error = _validateImage(uploadedFile)
    if error:
        st.error(f"❌ {error}")
        st.stop()

    # ── Preview ──────────────────────────────────────────────────────────
    st.image(image,
             caption=f"📷 {uploadedFile.name} ({image.size[0]}×{image.size[1]} px)",
             width=360)

    # ── Auto-analyze ─────────────────────────────────────────────────────
    # Tạo cache key từ file content + settings để tránh chạy lại khi rerun
    cacheKey = (uploadedFile.name, uploadedFile.size, topK, showGradCAM, gradcamAlpha)

    if st.session_state.get("_lastCacheKey") != cacheKey:
        with st.spinner("🔍 Đang phân tích ảnh..."):
            predictions, gradcamOverlay, errorMsg = _runAnalysis(
                pipeline, image, topK, showGradCAM, gradcamAlpha
            )

        if errorMsg and errorMsg != "gradcam_fallback":
            st.error("❌ Lỗi khi phân tích ảnh. Vui lòng thử lại hoặc dùng ảnh khác.")
            logger.error("Analysis error: %s", errorMsg)
            st.stop()

        st.session_state["results"] = {
            "predictions": predictions,
            "gradcamOverlay": gradcamOverlay,
            "gradcamFallback": errorMsg == "gradcam_fallback",
            "showGradCAM": showGradCAM,
        }
        st.session_state["_lastCacheKey"] = cacheKey

    # ── Display Results ──────────────────────────────────────────────────
    if "results" not in st.session_state:
        st.stop()

    res = st.session_state["results"]
    predictions = res["predictions"]
    gradcamOverlay = res.get("gradcamOverlay")

    if not predictions:
        st.stop()

    st.divider()

    # 1) Warnings (show first if confidence is low)
    _renderWarnings(predictions, imageWarnings)

    # 2) Prediction results
    st.markdown('<p class="section-header">📊 Kết quả phân tích</p>',
                unsafe_allow_html=True)
    _renderPredictionResults(predictions)

    # 3) Grad-CAM
    if res.get("showGradCAM") or gradcamOverlay is not None:
        _renderGradcamSection(
            image, gradcamOverlay,
            res.get("gradcamFallback", False),
            res.get("showGradCAM", False),
        )

    # 4) Recommendation
    _renderRecommendation(predictions[0]["className"])

    # 5) Footer
    _renderFooter(pipeline)


if __name__ == "__main__":
    main()
