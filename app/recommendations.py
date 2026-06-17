# app/recommendations.py
"""
Structured disease recommendations for PlantDocAI.

Mỗi entry chứa thông tin có cấu trúc về bệnh/trạng thái, được hiển thị
trong app Streamlit. Nội dung mang tính tham khảo, KHÔNG thay thế chuyên gia.

Keys khớp chính xác với classNames trong config.json artifact.
"""

from typing import Dict, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Structured Recommendation Type
# ─────────────────────────────────────────────────────────────────────────────
# Mỗi entry là dict với các key:
#   name        : Tên bệnh/trạng thái tiếng Việt
#   symptoms    : Mô tả triệu chứng quan sát được
#   treatment   : Gợi ý xử lý (mang tính tham khảo)
#   prevention  : Biện pháp phòng ngừa
#   notes       : Lưu ý bổ sung (optional)
#   isHealthy   : True nếu đây là trạng thái khỏe mạnh

RECOMMENDATIONS: Dict[str, dict] = {
    # ── Apple ─────────────────────────────────────────────────────────────────
    "Apple___Apple_scab": {
        "name": "Bệnh ghẻ táo (Apple Scab)",
        "symptoms": "Đốm xám-nâu hoặc xanh ô-liu trên mặt lá, lá có thể biến dạng hoặc rụng sớm.",
        "treatment": "Loại bỏ lá bị nhiễm và tiêu hủy. Tham khảo chuyên gia về thuốc bảo vệ thực vật phù hợp.",
        "prevention": "Vệ sinh vườn tốt, thu dọn lá rụng. Chọn giống kháng bệnh nếu có.",
        "notes": "Bệnh phát triển mạnh trong điều kiện ẩm ướt và mát.",
        "isHealthy": False,
    },
    "Apple___Black_rot": {
        "name": "Bệnh thối đen táo (Black Rot)",
        "symptoms": "Đốm nâu tím trên lá, lan rộng thành vòng đồng tâm. Quả có vùng thối đen.",
        "treatment": "Loại bỏ quả và cành bị nhiễm. Cắt tỉa cành chết và bị bệnh.",
        "prevention": "Đảm bảo vệ sinh vườn tốt, loại bỏ tàn dư thực vật cuối mùa.",
        "isHealthy": False,
    },
    "Apple___Cedar_apple_rust": {
        "name": "Bệnh rỉ sắt táo (Cedar Apple Rust)",
        "symptoms": "Đốm vàng-cam sáng trên mặt trên lá, mặt dưới có cấu trúc dạng sợi.",
        "treatment": "Tham khảo chuyên gia về thuốc bảo vệ thực vật phù hợp.",
        "prevention": "Tránh trồng gần cây tuyết tùng (cedar) — vật chủ trung gian của bệnh.",
        "isHealthy": False,
    },
    "Apple___healthy": {
        "name": "Lá táo khỏe mạnh",
        "symptoms": "Lá xanh đều, không có đốm hoặc biến dạng.",
        "treatment": "Tiếp tục chăm sóc và theo dõi định kỳ.",
        "prevention": "Duy trì vệ sinh vườn, tưới nước và bón phân đúng cách.",
        "isHealthy": True,
    },

    # ── Blueberry ─────────────────────────────────────────────────────────────
    "Blueberry___healthy": {
        "name": "Lá việt quất khỏe mạnh",
        "symptoms": "Lá xanh đều, không có dấu hiệu bệnh.",
        "treatment": "Tiếp tục chăm sóc và theo dõi định kỳ.",
        "prevention": "Đảm bảo đất có pH phù hợp (4.5–5.5) và thoát nước tốt.",
        "isHealthy": True,
    },

    # ── Cherry ────────────────────────────────────────────────────────────────
    "Cherry_(including_sour)___Powdery_mildew": {
        "name": "Bệnh phấn trắng anh đào (Powdery Mildew)",
        "symptoms": "Lớp phấn trắng trên bề mặt lá, lá có thể cong hoặc biến dạng.",
        "treatment": "Tham khảo chuyên gia về biện pháp xử lý phù hợp.",
        "prevention": "Đảm bảo thông thoáng gió giữa các cây, tránh tưới lên lá.",
        "isHealthy": False,
    },
    "Cherry_(including_sour)___healthy": {
        "name": "Lá anh đào khỏe mạnh",
        "symptoms": "Lá xanh đều, không có đốm hoặc lớp phấn.",
        "treatment": "Tiếp tục chăm sóc và theo dõi định kỳ.",
        "prevention": "Cắt tỉa định kỳ để đảm bảo thông gió.",
        "isHealthy": True,
    },

    # ── Corn ──────────────────────────────────────────────────────────────────
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "name": "Bệnh đốm xám lá ngô (Gray Leaf Spot)",
        "symptoms": "Đốm hình chữ nhật dài, màu xám-nâu, song song với gân lá.",
        "treatment": "Sử dụng giống kháng bệnh nếu có. Tham khảo chuyên gia về thuốc trừ nấm.",
        "prevention": "Luân canh cây trồng và quản lý tàn dư thực vật.",
        "isHealthy": False,
    },
    "Corn_(maize)___Common_rust_": {
        "name": "Bệnh rỉ sắt ngô (Common Rust)",
        "symptoms": "Đốm nhỏ màu nâu đỏ (pustule) phân bố trên cả hai mặt lá.",
        "treatment": "Theo dõi mức độ lây lan. Tham khảo chuyên gia về giống kháng bệnh.",
        "prevention": "Trồng giống kháng bệnh, theo dõi thời tiết ẩm ướt.",
        "isHealthy": False,
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "name": "Bệnh cháy lá phía bắc (Northern Leaf Blight)",
        "symptoms": "Đốm dài hình bầu dục (cigar-shaped), màu xám-xanh, có thể lan rộng.",
        "treatment": "Sử dụng giống kháng bệnh. Tham khảo chuyên gia về thuốc trừ nấm.",
        "prevention": "Luân canh cây trồng, quản lý tàn dư thực vật sau thu hoạch.",
        "isHealthy": False,
    },
    "Corn_(maize)___healthy": {
        "name": "Lá ngô khỏe mạnh",
        "symptoms": "Lá xanh đều, gân lá rõ, không có đốm.",
        "treatment": "Tiếp tục chăm sóc và theo dõi định kỳ.",
        "prevention": "Bón phân cân đối, đảm bảo tưới nước đúng cách.",
        "isHealthy": True,
    },

    # ── Grape ─────────────────────────────────────────────────────────────────
    "Grape___Black_rot": {
        "name": "Bệnh thối đen nho (Black Rot)",
        "symptoms": "Đốm nâu tròn trên lá với viền đậm. Quả bị thối và khô đen.",
        "treatment": "Loại bỏ quả và lá bị nhiễm. Đảm bảo vệ sinh vườn.",
        "prevention": "Cắt tỉa để thông thoáng, loại bỏ tàn dư bệnh.",
        "isHealthy": False,
    },
    "Grape___Esca_(Black_Measles)": {
        "name": "Bệnh Esca nho (Black Measles)",
        "symptoms": "Sọc đổi màu giữa các gân lá (tiger stripe pattern). Quả có đốm nhỏ tím.",
        "treatment": "Bệnh phức tạp, cần tham khảo chuyên gia nông nghiệp chuyên sâu.",
        "prevention": "Tránh tạo vết thương khi cắt tỉa, sử dụng kỹ thuật cắt tỉa đúng cách.",
        "notes": "Bệnh do phức hợp nấm gây ra, khó kiểm soát hoàn toàn.",
        "isHealthy": False,
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "name": "Bệnh cháy lá nho (Leaf Blight)",
        "symptoms": "Đốm nâu không đều trên lá, có thể lan rộng và gây khô lá.",
        "treatment": "Loại bỏ lá bị nhiễm, cải thiện thông gió trong vườn.",
        "prevention": "Tránh tưới lên lá, đảm bảo khoảng cách trồng phù hợp.",
        "isHealthy": False,
    },
    "Grape___healthy": {
        "name": "Lá nho khỏe mạnh",
        "symptoms": "Lá xanh đều, không có đốm hoặc biến dạng.",
        "treatment": "Tiếp tục chăm sóc và theo dõi định kỳ.",
        "prevention": "Cắt tỉa định kỳ, đảm bảo thông gió tốt.",
        "isHealthy": True,
    },

    # ── Orange ────────────────────────────────────────────────────────────────
    "Orange___Haunglongbing_(Citrus_greening)": {
        "name": "Bệnh vàng lá gân xanh cam (Citrus Greening / HLB)",
        "symptoms": "Lá vàng không đều (đốm loang lổ), quả nhỏ và méo, vị đắng.",
        "treatment": "Bệnh nghiêm trọng, chưa có cách chữa khỏi hoàn toàn. Cần báo cáo cơ quan nông nghiệp.",
        "prevention": "Kiểm soát rầy chổng cánh (vector truyền bệnh), sử dụng cây giống sạch bệnh.",
        "notes": "Đây là một trong những bệnh nghiêm trọng nhất của cây có múi trên toàn cầu.",
        "isHealthy": False,
    },

    # ── Peach ─────────────────────────────────────────────────────────────────
    "Peach___Bacterial_spot": {
        "name": "Bệnh đốm vi khuẩn đào (Bacterial Spot)",
        "symptoms": "Đốm nhỏ ngấm nước trên lá, sau chuyển nâu. Lá có thể rụng sớm.",
        "treatment": "Tham khảo chuyên gia về biện pháp phòng trừ vi khuẩn.",
        "prevention": "Sử dụng giống kháng bệnh. Tránh tưới phun lên tán lá.",
        "isHealthy": False,
    },
    "Peach___healthy": {
        "name": "Lá đào khỏe mạnh",
        "symptoms": "Lá xanh đều, hình dáng bình thường.",
        "treatment": "Tiếp tục chăm sóc và theo dõi định kỳ.",
        "prevention": "Cắt tỉa hợp lý, bón phân cân đối.",
        "isHealthy": True,
    },

    # ── Pepper ────────────────────────────────────────────────────────────────
    "Pepper,_bell___Bacterial_spot": {
        "name": "Bệnh đốm vi khuẩn ớt chuông (Bacterial Spot)",
        "symptoms": "Đốm nhỏ ngấm nước trên lá, lan rộng thành đốm nâu không đều.",
        "treatment": "Loại bỏ cây bệnh nặng. Tham khảo chuyên gia về thuốc gốc đồng.",
        "prevention": "Sử dụng hạt giống sạch bệnh, luân canh cây trồng.",
        "isHealthy": False,
    },
    "Pepper,_bell___healthy": {
        "name": "Lá ớt chuông khỏe mạnh",
        "symptoms": "Lá xanh đậm, bóng, không có đốm.",
        "treatment": "Tiếp tục chăm sóc và theo dõi định kỳ.",
        "prevention": "Tưới nước đều, bón phân cân đối.",
        "isHealthy": True,
    },

    # ── Potato ────────────────────────────────────────────────────────────────
    "Potato___Early_blight": {
        "name": "Bệnh cháy sớm khoai tây (Early Blight)",
        "symptoms": "Đốm nâu hình đồng tâm (target-like) trên lá già, lan dần lên trên.",
        "treatment": "Loại bỏ lá bị nhiễm. Tham khảo chuyên gia về thuốc trừ nấm.",
        "prevention": "Luân canh cây trồng, loại bỏ tàn dư thực vật, tưới gốc.",
        "isHealthy": False,
    },
    "Potato___Late_blight": {
        "name": "Bệnh mốc sương khoai tây (Late Blight)",
        "symptoms": "Đốm nước lớn, nâu-đen trên lá, mặt dưới lá có lớp mốc trắng. Lan nhanh.",
        "treatment": "Bệnh nguy hiểm, cần xử lý kịp thời. Tham khảo chuyên gia ngay.",
        "prevention": "Sử dụng giống kháng, theo dõi thời tiết ẩm, quản lý tưới nước.",
        "notes": "Bệnh có thể lan rất nhanh trong điều kiện mát và ẩm.",
        "isHealthy": False,
    },
    "Potato___healthy": {
        "name": "Lá khoai tây khỏe mạnh",
        "symptoms": "Lá xanh đều, phát triển bình thường.",
        "treatment": "Tiếp tục chăm sóc và theo dõi định kỳ.",
        "prevention": "Luân canh cây trồng, bón phân cân đối.",
        "isHealthy": True,
    },

    # ── Raspberry ─────────────────────────────────────────────────────────────
    "Raspberry___healthy": {
        "name": "Lá mâm xôi khỏe mạnh",
        "symptoms": "Lá xanh đều, không có dấu hiệu bệnh.",
        "treatment": "Tiếp tục chăm sóc và theo dõi định kỳ.",
        "prevention": "Cắt tỉa cành già, đảm bảo thoát nước tốt.",
        "isHealthy": True,
    },

    # ── Soybean ───────────────────────────────────────────────────────────────
    "Soybean___healthy": {
        "name": "Lá đậu nành khỏe mạnh",
        "symptoms": "Lá xanh đều, hình dạng bình thường.",
        "treatment": "Tiếp tục chăm sóc và theo dõi định kỳ.",
        "prevention": "Luân canh cây trồng, kiểm tra sâu bệnh định kỳ.",
        "isHealthy": True,
    },

    # ── Squash ────────────────────────────────────────────────────────────────
    "Squash___Powdery_mildew": {
        "name": "Bệnh phấn trắng bí (Powdery Mildew)",
        "symptoms": "Lớp bột trắng trên bề mặt lá, lan rộng dần. Lá có thể vàng và khô.",
        "treatment": "Tham khảo chuyên gia về thuốc trừ nấm phù hợp.",
        "prevention": "Cải thiện thông gió, tránh tưới lên lá, trồng giống kháng.",
        "isHealthy": False,
    },

    # ── Strawberry ────────────────────────────────────────────────────────────
    "Strawberry___Leaf_scorch": {
        "name": "Bệnh cháy lá dâu tây (Leaf Scorch)",
        "symptoms": "Đốm tím nhỏ trên lá, lan rộng và làm khô mép lá. Lá chuyển nâu.",
        "treatment": "Loại bỏ lá bị nhiễm, đảm bảo thoát nước tốt.",
        "prevention": "Trồng với khoảng cách phù hợp, tránh tưới quá nhiều.",
        "isHealthy": False,
    },
    "Strawberry___healthy": {
        "name": "Lá dâu tây khỏe mạnh",
        "symptoms": "Lá xanh tươi, 3 lá chét đều, không có đốm.",
        "treatment": "Tiếp tục chăm sóc và theo dõi định kỳ.",
        "prevention": "Tưới nước đều, bón phân phù hợp giai đoạn sinh trưởng.",
        "isHealthy": True,
    },

    # ── Tomato ────────────────────────────────────────────────────────────────
    "Tomato___Bacterial_spot": {
        "name": "Bệnh đốm vi khuẩn cà chua (Bacterial Spot)",
        "symptoms": "Đốm nhỏ ngấm nước trên lá, sau chuyển nâu-đen với viền vàng.",
        "treatment": "Loại bỏ lá bệnh nặng. Tham khảo chuyên gia về thuốc gốc đồng.",
        "prevention": "Sử dụng hạt giống sạch bệnh, tránh tưới trên lá.",
        "isHealthy": False,
    },
    "Tomato___Early_blight": {
        "name": "Bệnh cháy sớm cà chua (Early Blight)",
        "symptoms": "Đốm nâu hình đồng tâm trên lá già, viền vàng. Lan từ dưới lên.",
        "treatment": "Loại bỏ lá bị nhiễm từ dưới lên. Tham khảo chuyên gia về thuốc trừ nấm.",
        "prevention": "Luân canh cây trồng, loại bỏ tàn dư thực vật, che phủ đất.",
        "isHealthy": False,
    },
    "Tomato___Late_blight": {
        "name": "Bệnh mốc sương cà chua (Late Blight)",
        "symptoms": "Đốm nước lớn, xám-xanh trên lá, lan nhanh. Mặt dưới có mốc trắng.",
        "treatment": "Bệnh nguy hiểm, cần xử lý kịp thời. Tham khảo chuyên gia ngay.",
        "prevention": "Theo dõi thời tiết, tránh tưới buổi tối, đảm bảo thông gió.",
        "notes": "Có thể lan rất nhanh và gây thiệt hại lớn nếu không xử lý sớm.",
        "isHealthy": False,
    },
    "Tomato___Leaf_Mold": {
        "name": "Bệnh mốc lá cà chua (Leaf Mold)",
        "symptoms": "Đốm vàng mặt trên lá, mốc xám-tím mặt dưới lá. Thường ở lá dưới.",
        "treatment": "Cải thiện thông gió, giảm độ ẩm. Tham khảo chuyên gia.",
        "prevention": "Thông gió tốt trong nhà kính, tránh ẩm độ quá cao.",
        "isHealthy": False,
    },
    "Tomato___Septoria_leaf_spot": {
        "name": "Bệnh đốm lá Septoria cà chua",
        "symptoms": "Đốm tròn nhỏ với tâm xám và viền nâu-đen. Bắt đầu từ lá dưới.",
        "treatment": "Loại bỏ lá bị nhiễm từ dưới lên. Tham khảo chuyên gia.",
        "prevention": "Che phủ đất, tưới gốc, luân canh cây trồng.",
        "isHealthy": False,
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "name": "Nhện đỏ hai chấm trên cà chua (Spider Mites)",
        "symptoms": "Đốm vàng nhỏ li ti trên mặt trên lá, mạng nhện mặt dưới lá.",
        "treatment": "Tham khảo biện pháp sinh học (thiên địch) hoặc thuốc trừ nhện phù hợp.",
        "prevention": "Kiểm tra mặt dưới lá thường xuyên, duy trì độ ẩm phù hợp.",
        "isHealthy": False,
    },
    "Tomato___Target_Spot": {
        "name": "Bệnh đốm vòng cà chua (Target Spot)",
        "symptoms": "Đốm nâu hình đồng tâm trên lá, tương tự Early Blight nhưng nhỏ hơn.",
        "treatment": "Loại bỏ lá bệnh, cải thiện thoát nước. Tham khảo chuyên gia.",
        "prevention": "Luân canh cây trồng, cải thiện thoát nước, tránh trồng quá dày.",
        "isHealthy": False,
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name": "Virus xoăn vàng lá cà chua (TYLCV)",
        "symptoms": "Lá cong lên, vàng viền lá, cây lùn, hoa rụng. Lá nhỏ và dày.",
        "treatment": "Không có thuốc chữa virus. Loại bỏ cây bệnh để tránh lây lan.",
        "prevention": "Kiểm soát bọ phấn trắng (vector), sử dụng giống kháng, lưới chắn côn trùng.",
        "notes": "Bệnh do virus, lây truyền qua bọ phấn trắng (whitefly).",
        "isHealthy": False,
    },
    "Tomato___Tomato_mosaic_virus": {
        "name": "Virus khảm cà chua (Tomato Mosaic Virus)",
        "symptoms": "Lá có vết loang lổ xanh đậm-nhạt (mosaic pattern), lá biến dạng.",
        "treatment": "Không có thuốc chữa virus. Loại bỏ cây bệnh, vệ sinh dụng cụ.",
        "prevention": "Vệ sinh tay và dụng cụ khi làm vườn, sử dụng hạt giống sạch bệnh.",
        "notes": "Virus rất bền, có thể lây qua tiếp xúc cơ học.",
        "isHealthy": False,
    },
    "Tomato___healthy": {
        "name": "Lá cà chua khỏe mạnh",
        "symptoms": "Lá xanh đều, phát triển bình thường, không có đốm hay biến dạng.",
        "treatment": "Tiếp tục chăm sóc và theo dõi định kỳ.",
        "prevention": "Tưới nước đều, bón phân cân đối, theo dõi sâu bệnh.",
        "isHealthy": True,
    },
}


def getRecommendation(className: str) -> Optional[dict]:
    """Lấy recommendation cho class name. Trả về None nếu không có."""
    return RECOMMENDATIONS.get(className)


def getDiseaseName(className: str) -> str:
    """Lấy tên bệnh tiếng Việt. Fallback về class name gốc."""
    rec = RECOMMENDATIONS.get(className)
    if rec:
        return rec["name"]
    return className.replace("___", " — ").replace("_", " ")
