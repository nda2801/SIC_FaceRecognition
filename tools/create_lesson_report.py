from pathlib import Path
import math
import textwrap

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_ROOT / "outputs" / "report_assets"
OUTPUT_PATH = PROJECT_ROOT / "Tong_hop_Custom_Vision_Transformer_SIC.docx"
ASSET_DIR.mkdir(parents=True, exist_ok=True)

FONT_REGULAR = Path("C:/Windows/Fonts/arial.ttf")
FONT_BOLD = Path("C:/Windows/Fonts/arialbd.ttf")


def font(size, bold=False):
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(str(path), size=size)


def draw_centered_text(draw, box, text, fill, size=28, bold=False):
    x1, y1, x2, y2 = box
    fnt = font(size, bold)
    max_chars = max(8, int((x2 - x1) / (size * 0.58)))
    wrapped = "\n".join(textwrap.wrap(text, width=max_chars))
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=fnt, spacing=7, align="center")
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.multiline_text(
        ((x1 + x2 - width) / 2, (y1 + y2 - height) / 2),
        wrapped,
        font=fnt,
        fill=fill,
        spacing=7,
        align="center",
    )


def box(draw, coords, text, color, text_color="#10233f", size=26):
    draw.rounded_rectangle(coords, radius=22, fill=color, outline="#355070", width=3)
    draw_centered_text(draw, coords, text, text_color, size=size, bold=True)


def arrow(draw, start, end, color="#355070", width=6):
    draw.line([start, end], fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 18
    for offset in (2.55, -2.55):
        point = (
            end[0] + length * math.cos(angle + offset),
            end[1] + length * math.sin(angle + offset),
        )
        draw.line([end, point], fill=color, width=width)


def save_pipeline_diagram():
    path = ASSET_DIR / "pipeline.png"
    image = Image.new("RGB", (1800, 470), "white")
    draw = ImageDraw.Draw(image)
    items = [
        ("Dataset Pins\n17.534 ảnh / 105 người", "#d8f3dc"),
        ("data.py\nĐọc, chia tập, augmentation", "#caf0f8"),
        ("DataLoader\nBatch [16,3,112,112]", "#ade8f4"),
        ("Custom ViT\n1.366.185 tham số", "#ffe5b4"),
        ("Logits\n[16,105]", "#ffd6e0"),
    ]
    width, gap, y1, y2 = 285, 70, 115, 345
    x = 45
    for index, (label, color) in enumerate(items):
        coords = (x, y1, x + width, y2)
        box(draw, coords, label, color, size=25)
        if index < len(items) - 1:
            arrow(draw, (x + width, 230), (x + width + gap - 8, 230))
        x += width + gap
    draw.text((45, 28), "LUỒNG DỮ LIỆU HIỆN TẠI", font=font(34, True), fill="#10233f")
    image.save(path)
    return path


def save_vit_diagram():
    path = ASSET_DIR / "vit_shapes.png"
    image = Image.new("RGB", (1550, 1230), "white")
    draw = ImageDraw.Draw(image)
    draw.text((55, 30), "CUSTOM VISION TRANSFORMER – LUỒNG SHAPE", font=font(36, True), fill="#10233f")
    items = [
        ("Batch ảnh\n[16, 3, 112, 112]", "#d8f3dc"),
        ("Patch Embedding\nConv2d kernel=16, stride=16", "#caf0f8"),
        ("49 patch tokens\n[16, 49, 192]", "#ade8f4"),
        ("Thêm CLS + vị trí\n[16, 50, 192]", "#ffe5b4"),
        ("4 Encoder Blocks\nshape giữ nguyên", "#ffcad4"),
        ("Lấy CLS\n[16, 192]", "#e2d1f9"),
        ("Linear(192,105)\nLogits [16,105]", "#caffbf"),
    ]
    y = 105
    for index, (label, color) in enumerate(items):
        coords = (360, y, 1190, y + 120)
        box(draw, coords, label, color, size=27)
        if index < len(items) - 1:
            arrow(draw, (775, y + 120), (775, y + 162))
        y += 165
    image.save(path)
    return path


def save_block_diagram():
    path = ASSET_DIR / "transformer_block.png"
    image = Image.new("RGB", (1700, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((50, 28), "MỘT TRANSFORMER ENCODER BLOCK", font=font(36, True), fill="#10233f")

    labels = [
        ("x\n[B,T,E]", "#d8f3dc"),
        ("LayerNorm\n(norm_attention)", "#caf0f8"),
        ("Multi-Head\nSelf-Attention", "#ffe5b4"),
        ("Cộng residual\nx + attention", "#ffd6e0"),
        ("LayerNorm\n(norm_mlp)", "#caf0f8"),
        ("FeedForward\nE → 2E → E", "#e2d1f9"),
        ("Cộng residual\nx + mlp", "#ffd6e0"),
        ("Output\n[B,T,E]", "#caffbf"),
    ]
    width, gap, y1, y2 = 185, 24, 335, 565
    x = 35
    centers = []
    for label, color in labels:
        coords = (x, y1, x + width, y2)
        box(draw, coords, label, color, size=22)
        centers.append((x + width / 2, (y1 + y2) / 2))
        x += width + gap
    for i in range(len(centers) - 1):
        arrow(draw, (centers[i][0] + width / 2, centers[i][1]), (centers[i + 1][0] - width / 2 - 6, centers[i + 1][1]), width=5)
    draw.text((80, 680), "Pre-Norm: chuẩn hóa trước Attention/MLP", font=font(28, True), fill="#355070")
    draw.text((80, 735), "Residual: giữ thông tin cũ và cộng thông tin mới", font=font(28, True), fill="#355070")
    image.save(path)
    return path


def save_attention_diagram():
    path = ASSET_DIR / "attention.png"
    image = Image.new("RGB", (1650, 950), "white")
    draw = ImageDraw.Draw(image)
    draw.text((45, 30), "SELF-ATTENTION: QUERY – KEY – VALUE", font=font(36, True), fill="#10233f")
    box(draw, (80, 160, 420, 355), "Cùng tập token\nCLS + 49 patch", "#d8f3dc", size=27)
    branches = [
        ((580, 100, 930, 260), "Linear Q\nQuery: đang tìm gì?", "#caf0f8"),
        ((580, 390, 930, 550), "Linear K\nKey: có phù hợp?", "#ffe5b4"),
        ((580, 680, 930, 840), "Linear V\nValue: nội dung", "#e2d1f9"),
    ]
    for coords, label, color in branches:
        box(draw, coords, label, color, size=25)
        arrow(draw, (420, 255), (coords[0] - 8, (coords[1] + coords[3]) / 2), width=5)
    box(draw, (1110, 250, 1570, 545), "So sánh Q với K\n→ trọng số chú ý\n→ tổng hợp Value", "#ffd6e0", size=29)
    arrow(draw, (930, 180), (1100, 340), width=5)
    arrow(draw, (930, 470), (1100, 400), width=5)
    arrow(draw, (930, 760), (1100, 470), width=5)
    draw.text((1050, 660), "3 heads: 192 / 3 = 64 chiều/head", font=font(28, True), fill="#355070")
    draw.text((1050, 715), "Ghép 3 heads → trở lại 192 chiều", font=font(28, True), fill="#355070")
    image.save(path)
    return path


def save_comparison_chart():
    path = ASSET_DIR / "model_parameters.png"
    models = [
        ("Custom ViT", 1.366, "#2a9d8f"),
        ("MobileNetV2", 3.5, "#52b788"),
        ("ResNet18", 11.7, "#457b9d"),
        ("ResNeSt50", 27.5, "#f4a261"),
        ("AlexNet", 61.1, "#e76f51"),
        ("ViT-B/16", 86.6, "#9d4edd"),
    ]
    image = Image.new("RGB", (1650, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((55, 28), "SO SÁNH SỐ THAM SỐ (TRIỆU, XẤP XỈ)", font=font(36, True), fill="#10233f")
    max_value = max(value for _, value, _ in models)
    x0, max_width, y = 390, 1090, 135
    for name, value, color in models:
        draw.text((55, y + 16), name, font=font(27, True), fill="#10233f")
        width = max(20, int(max_width * value / max_value))
        draw.rounded_rectangle((x0, y, x0 + width, y + 65), radius=15, fill=color)
        draw.text((x0 + width + 18, y + 14), f"{value:.3g}M", font=font(25, True), fill="#10233f")
        y += 112
    draw.text((55, 825), "Lưu ý: số liệu model chuẩn phụ thuộc classifier/cấu hình; dùng để định hướng, không thay thế benchmark cùng điều kiện.", font=font(22), fill="#6c757d")
    image.save(path)
    return path


def save_transformer_chart():
    path = ASSET_DIR / "transformer_family.png"
    models = [
        ("SIC-ViT-4 hiện tại", 1.366, "#2a9d8f"),
        ("MobileViT-XS", 2.3, "#52b788"),
        ("ViT-Tiny-inspired", 5.516, "#457b9d"),
        ("DeiT-Tiny", 5.7, "#4cc9f0"),
        ("ViT-Small/16-inspired", 21.65, "#f4a261"),
        ("Swin-T", 28.3, "#e76f51"),
        ("ViT-B/16", 86.6, "#9d4edd"),
    ]
    image = Image.new("RGB", (1650, 1050), "white")
    draw = ImageDraw.Draw(image)
    draw.text((55, 28), "SO SÁNH CÁC KIẾN TRÚC TRANSFORMER", font=font(36, True), fill="#10233f")
    max_value = max(value for _, value, _ in models)
    x0, max_width, y = 445, 1010, 125
    for name, value, color in models:
        draw.text((55, y + 12), name, font=font(25, True), fill="#10233f")
        width = max(20, int(max_width * value / max_value))
        draw.rounded_rectangle((x0, y, x0 + width, y + 62), radius=14, fill=color)
        draw.text((x0 + width + 18, y + 12), f"{value:.3g}M", font=font(23, True), fill="#10233f")
        y += 117
    draw.text((55, 940), "SIC-ViT-4 và ViT-Tiny-inspired là cấu hình tự cài đặt trong project; các số liệu còn lại là kiến trúc chuẩn xấp xỉ.", font=font(21), fill="#6c757d")
    image.save(path)
    return path


def save_deployment_diagram():
    path = ASSET_DIR / "deployment.png"
    image = Image.new("RGB", (1700, 560), "white")
    draw = ImageDraw.Draw(image)
    draw.text((50, 25), "HƯỚNG TRIỂN KHAI SẢN PHẨM", font=font(36, True), fill="#10233f")
    box(draw, (65, 170, 390, 390), "PyTorch\nTrain Custom ViT", "#caf0f8", size=28)
    arrow(draw, (390, 280), (540, 280))
    box(draw, (550, 170, 870, 390), "Export ONNX\nmodel + preprocessing", "#ffe5b4", size=28)
    branches = [
        ((1035, 80, 1600, 205), "Web: FastAPI / ONNX Runtime Web", "#d8f3dc"),
        ((1035, 220, 1600, 345), "Mobile: ONNX Runtime Mobile / ExecuTorch", "#ffd6e0"),
        ((1035, 360, 1600, 485), "WinForms: ONNX Runtime C#", "#e2d1f9"),
    ]
    for coords, label, color in branches:
        box(draw, coords, label, color, size=24)
        arrow(draw, (870, 280), (coords[0] - 10, (coords[1] + coords[3]) / 2), width=5)
    image.save(path)
    return path


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("Trang ")
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = "PAGE"
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char1)
    run._r.append(instr_text)
    run._r.append(fld_char2)


def style_document(document):
    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15
    for style_name, size, color in [
        ("Title", 26, "12355B"),
        ("Heading 1", 18, "12355B"),
        ("Heading 2", 14, "1D5D75"),
        ("Heading 3", 12, "2A6F97"),
    ]:
        style = styles[style_name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)
        style.font.bold = True


def add_code(document, text):
    paragraph = document.add_paragraph()
    paragraph.style = document.styles["Normal"]
    paragraph.paragraph_format.left_indent = Cm(0.7)
    paragraph.paragraph_format.right_indent = Cm(0.7)
    paragraph.paragraph_format.space_before = Pt(3)
    paragraph.paragraph_format.space_after = Pt(8)
    p_pr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), "F1F5F9")
    p_pr.append(shd)
    run = paragraph.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(9.5)
    return paragraph


def add_table(document, headers, rows, widths=None):
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    header_cells = table.rows[0].cells
    for i, header in enumerate(headers):
        header_cells[i].text = header
        set_cell_shading(header_cells[i], "D9EAF7")
        header_cells[i].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for run in header_cells[i].paragraphs[0].runs:
            run.bold = True
            run.font.name = "Arial"
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = str(value)
            cells[i].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for paragraph in cells[i].paragraphs:
                for run in paragraph.runs:
                    run.font.name = "Arial"
                    run.font.size = Pt(9.5)
    if widths:
        for row in table.rows:
            for i, width in enumerate(widths):
                row.cells[i].width = Cm(width)
    document.add_paragraph()
    return table


def build_document():
    pipeline = save_pipeline_diagram()
    vit = save_vit_diagram()
    block = save_block_diagram()
    attention = save_attention_diagram()
    comparison = save_comparison_chart()
    transformer_chart = save_transformer_chart()
    deployment = save_deployment_diagram()

    doc = Document()
    style_document(doc)
    section = doc.sections[0]
    section.top_margin = Cm(1.8)
    section.bottom_margin = Cm(1.7)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)
    add_page_number(section.footer.paragraphs[0])

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("TỔNG HỢP CUSTOM VISION TRANSFORMER\nCHO NHẬN DIỆN KHUÔN MẶT")
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("Samsung Innovation Campus – Ghi chú học tập và thiết kế mô hình")
    run.bold = True
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(53, 80, 112)
    info = doc.add_paragraph()
    info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info.add_run("Ngày tổng hợp: 23/07/2026\n")
    info.add_run("Dataset: Pins Face Recognition – 105 danh tính, 17.534 ảnh\n")
    info.add_run("Model hiện tại: Custom ViT, train từ đầu, không dùng pretrained weights")
    doc.add_picture(str(pipeline), width=Inches(6.7))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break()

    doc.add_heading("1. Mục tiêu và trạng thái hiện tại", level=1)
    doc.add_paragraph(
        "Mục tiêu trước mắt là hiểu và tự xây dựng một mô hình Vision Transformer nhận diện danh tính từ ảnh khuôn mặt đã crop. "
        "Face Detection nhiều khuôn mặt sẽ được tích hợp sau khi phần nhận diện một khuôn mặt được train và đánh giá ổn định."
    )
    add_table(
        doc,
        ["Hạng mục", "Kết quả hiện tại"],
        [
            ("Dataset", "105 người, 17.534 ảnh"),
            ("Train / Validation / Test", "12.225 / 2.574 / 2.735 ảnh"),
            ("Input batch", "[16, 3, 112, 112]"),
            ("Output logits", "[16, 105]"),
            ("Số tham số", "1.366.185"),
            ("Trạng thái", "DataLoader và forward pass đã chạy đúng; chưa training"),
        ],
        widths=[5, 11],
    )

    doc.add_heading("2. Mô hình hoạt động như thế nào?", level=1)
    doc.add_picture(str(vit), width=Inches(6.3))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("2.1. Patch Embedding", level=2)
    doc.add_paragraph(
        "Ảnh 112×112 được chia thành các patch 16×16. Có 7 patch theo mỗi chiều, tổng cộng 49 patch. "
        "Conv2d với kernel_size=16 và stride=16 vừa chia patch không chồng lấp, vừa chiếu mỗi patch thành vector 192 chiều."
    )
    add_code(doc, "[B, 3, 112, 112] → Conv2d(3,192,kernel=16,stride=16) → [B,192,7,7]\n→ flatten → [B,192,49] → transpose → [B,49,192]")

    doc.add_heading("2.2. CLS token và Positional Embedding", level=2)
    doc.add_paragraph(
        "CLS là một tham số học được, đóng vai trò vector tổng hợp cho toàn ảnh. Sau khi thêm CLS, 49 patch trở thành 50 token. "
        "Positional Embedding cung cấp địa chỉ cho mỗi token để Transformer phân biệt vùng trên, dưới, trái và phải của khuôn mặt."
    )
    add_code(doc, "49 patch + 1 CLS = 50 token\n[B,50,192] + position_embedding[1,50,192] → [B,50,192]")

    doc.add_heading("2.3. Transformer Encoder Block", level=2)
    doc.add_picture(str(block), width=Inches(6.7))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(
        "Mỗi block dùng cấu trúc Pre-Norm: LayerNorm được thực hiện trước Attention và trước FeedForward. "
        "Hai residual connection giữ thông tin cũ, hỗ trợ gradient và giúp nhiều block nối tiếp vẫn train ổn định."
    )

    doc.add_heading("2.4. Self-Attention và Multi-Head Attention", level=2)
    doc.add_picture(str(attention), width=Inches(6.6))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_table(
        doc,
        ["Thành phần", "Cách hiểu trực giác"],
        [
            ("Query", "Token đang muốn tìm loại thông tin nào?"),
            ("Key", "Token khác có thông tin phù hợp đến mức nào?"),
            ("Value", "Nội dung thực tế được tổng hợp theo trọng số chú ý"),
            ("Multi-head", "Nhiều phép Attention song song; 192 chiều chia thành 3 head × 64 chiều"),
        ],
        widths=[4, 12],
    )

    doc.add_heading("2.5. Classifier", level=2)
    doc.add_paragraph(
        "Sau bốn Encoder Block, model lấy token ở vị trí 0 (CLS), thu được vector [B,192]. "
        "Linear(192,105) tạo 105 logits cho mỗi ảnh. CrossEntropyLoss sẽ nhận logits trực tiếp; không đặt Softmax trong forward."
    )
    add_code(doc, "tokens[:,0] → class_features [B,192]\nLinear(192,105) → logits [B,105]")

    doc.add_page_break()
    doc.add_heading("3. Giải thích các file, hàm và lớp", level=1)

    doc.add_heading("3.1. config.py", level=2)
    doc.add_paragraph("File cấu hình dùng argparse để thay đổi tham số từ terminal mà không sửa mã nguồn.")
    add_table(
        doc,
        ["Tham số", "Mặc định", "Ý nghĩa"],
        [
            ("dataset_root", "dataset", "Thư mục chứa 105 thư mục danh tính"),
            ("image_size", "112", "Kích thước ảnh đầu vào"),
            ("train/val/test ratio", "0.70/0.15/0.15", "Tỷ lệ chia dữ liệu"),
            ("batch_size", "16", "Số ảnh trong một batch"),
            ("num_workers", "0", "Số tiến trình đọc dữ liệu; an toàn khi bắt đầu trên Windows"),
            ("lr", "3e-4", "Learning rate"),
            ("weight_decay", "1e-4", "Regularization cho trọng số"),
            ("epochs", "50", "Số epoch tối đa"),
            ("early_stop", "7", "Patience cho Early Stopping"),
            ("seed", "42", "Tái hiện cách chia và ngẫu nhiên"),
            ("patch_size", "16", "Kích thước mỗi patch"),
            ("embed_dim", "192", "Số đặc trưng mỗi token"),
            ("depth", "4", "Số Encoder Block"),
            ("num_heads", "3", "Số Attention head"),
            ("mlp_ratio", "2.0", "FeedForward 192→384→192"),
            ("dropout", "0.1", "Tỷ lệ Dropout"),
        ],
        widths=[4, 3, 9],
    )
    add_table(
        doc,
        ["Hàm/đoạn", "Vai trò"],
        [
            ("get_parser()", "Tạo ArgumentParser, đăng ký tham số, parse terminal, kiểm tra tỷ lệ và khả năng chia patch, rồi trả cfg."),
            ("if __name__ == '__main__'", "Cho phép chạy trực tiếp config.py để in cấu hình mà không tự chạy khi import."),
        ],
        widths=[5, 11],
    )

    doc.add_heading("3.2. data.py", level=2)
    add_table(
        doc,
        ["Hàm/lớp", "Đầu vào", "Đầu ra và vai trò"],
        [
            ("read_data", "root_path, tỷ lệ, seed", "Tìm thư mục người, gán label 0–104, lọc ảnh, shuffle và chia X/y cho train-val-test."),
            ("ImageDataset.__init__", "image_paths, labels, transform", "Lưu đường dẫn, nhãn và pipeline xử lý ảnh."),
            ("ImageDataset.__len__", "self", "Trả số mẫu để DataLoader biết độ dài dataset."),
            ("ImageDataset.__getitem__", "index", "Mở ảnh, convert RGB, transform, trả tensor ảnh và label."),
            ("get_transforms", "image_size", "Tạo train_transform có augmentation và eval_transform ổn định."),
            ("build_dataloaders", "cfg", "Tạo ba Dataset, ba DataLoader và trả class_names."),
        ],
        widths=[4.5, 4.2, 7.3],
    )
    doc.add_paragraph("Train transform:")
    add_code(doc, "Resize → RandomHorizontalFlip → RandomRotation → ColorJitter → ToTensor → Normalize")
    doc.add_paragraph("Validation/Test/App transform:")
    add_code(doc, "Resize → ToTensor → Normalize")
    doc.add_paragraph(
        "Với mean=std=0.5, pixel sau ToTensor từ [0,1] được đưa về gần [-1,1]. Web/Mobile phải dùng cùng preprocessing với lúc train."
    )

    doc.add_heading("3.3. model.py", level=2)
    add_table(
        doc,
        ["Lớp/hàm", "Vai trò", "Shape chính"],
        [
            ("PatchEmbedding", "Conv2d chia patch và tạo embedding.", "[B,3,112,112] → [B,49,192]"),
            ("FeedForward", "MLP xử lý riêng từng token với GELU và Dropout.", "[B,T,192] → [B,T,384] → [B,T,192]"),
            ("TransformerEncoderBlock", "LayerNorm, MultiheadAttention, MLP và hai residual.", "[B,T,192] → [B,T,192]"),
            ("VisionTransformer", "Ghép PatchEmbedding, CLS, vị trí, 4 block và classifier.", "[B,3,112,112] → [B,105]"),
            ("build_model", "Lấy số lớp từ dataset và hyperparameter từ cfg để tạo model.", "num_classes=len(class_names)"),
        ],
        widths=[4.5, 7.5, 4.5],
    )
    doc.add_heading("Các lớp PyTorch nền tảng", level=3)
    add_table(
        doc,
        ["Layer", "Ý nghĩa"],
        [
            ("nn.Conv2d", "Ở đây dùng kernel=stride=patch_size để tạo các patch không chồng lấp."),
            ("nn.Linear", "Biến đổi chiều đặc trưng; trọng số được học."),
            ("nn.GELU", "Activation phi tuyến thường dùng trong Transformer."),
            ("nn.Dropout", "Tắt ngẫu nhiên một phần giá trị khi train để giảm overfitting."),
            ("nn.LayerNorm", "Chuẩn hóa chiều đặc trưng của từng token."),
            ("nn.MultiheadAttention", "Tính Self-Attention theo nhiều head song song."),
            ("nn.Parameter", "Đăng ký CLS và Positional Embedding thành tham số được học."),
            ("nn.Sequential", "Ghép nhiều layer/block chạy tuần tự."),
        ],
        widths=[5, 11],
    )

    doc.add_page_break()
    doc.add_heading("4. So sánh với AlexNet, MobileNet và các kiến trúc khác", level=1)
    doc.add_picture(str(comparison), width=Inches(6.7))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_table(
        doc,
        ["Model", "Cơ chế chính", "Ưu điểm", "Hạn chế / phù hợp"],
        [
            ("AlexNet", "CNN convolution + pooling + FC lớn", "Dễ hiểu, liên hệ bài đã học", "Cũ, khoảng 61M tham số, FC nặng, không phù hợp mobile hiện đại"),
            ("MobileNetV2", "Depthwise separable convolution", "Nhẹ, nhanh, rất phù hợp thiết bị biên", "Khả năng biểu diễn thấp hơn model lớn; là baseline deployment tốt"),
            ("ResNet18", "CNN + residual connection", "Ổn định, dễ train, baseline mạnh", "Không phải Transformer; khoảng 11.7M tham số"),
            ("ResNeSt50", "CNN + split-attention", "Trích xuất đặc trưng mạnh, đúng góp ý ban đầu của giảng viên", "Nặng hơn, khoảng 27.5M tham số; cần benchmark GPU"),
            ("Custom ViT hiện tại", "Patch + Self-Attention", "Tự xây, chỉ 1.366M tham số, phù hợp học và GPU 4GB", "Train từ đầu trên dataset vừa có nguy cơ overfitting; chất lượng chưa biết trước khi train"),
            ("ViT-Tiny-inspired", "12 block, embedding 192, 3 heads", "Cùng hướng ViT nhưng nhiều tầng hơn; khoảng 5.52M tham số theo code hiện tại", "Tên này chỉ mô tả cấu hình lấy cảm hứng, không phải bản chuẩn vì input=112 và không distillation"),
            ("DeiT-Tiny", "ViT-Tiny + data-efficient training/distillation", "Cho thấy recipe huấn luyện quan trọng khi dữ liệu không lớn", "Không được dùng nguyên model/pretrained trong project này; dùng làm tài liệu tham khảo"),
            ("Swin-T", "Window attention + shifted windows, hierarchical", "Giảm chi phí Attention, nhiều scale, phù hợp vision/detection", "Nặng khoảng 28.3M; không phải bản nhóm đang tự cài"),
            ("ViT-B/16", "12 block, embedding 768, 12 heads", "Khả năng lớn khi có nhiều dữ liệu/pretraining", "Khoảng 86.6M tham số, quá nặng cho mục tiêu hiện tại"),
        ],
        widths=[3, 4, 4.5, 5],
    )
    doc.add_heading("4.1. Các Transformer cần phân biệt", level=2)
    doc.add_picture(str(transformer_chart), width=Inches(6.7))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(
        "Bản hiện tại nên được gọi là SIC-ViT-4: tên thí nghiệm của nhóm, không phải một kiến trúc đã công bố. "
        "Nó lấy các thành phần cốt lõi từ ViT gốc (patch projection, CLS, positional embedding, encoder attention/MLP) nhưng giảm số tầng, embedding và MLP để phù hợp GPU 4 GB. "
        "Cấu hình ViT-Tiny-inspired 12 block có cùng embedding 192 và 3 heads nhưng depth=12, mlp_ratio=4, tạo 5.516.457 tham số theo code hiện tại."
    )
    add_table(
        doc,
        ["Cấu hình thí nghiệm", "Depth", "Embed", "Heads", "MLP", "Tham số project"],
        [
            ("SIC-ViT-4", "4", "192", "3", "2×", "1.366.185"),
            ("ViT-Tiny-inspired", "12", "192", "3", "4×", "5.516.457"),
            ("ViT-Small-inspired", "12", "384", "6", "4×", "21.649.641 (nặng, chỉ tham khảo)"),
        ],
        widths=[4, 2, 2, 2, 2, 5],
    )
    doc.add_paragraph(
        "Để so sánh có cơ sở, hãy giữ nguyên dataset split, seed, image_size, augmentation, optimizer và tiêu chí Early Stopping; chỉ thay cấu hình model. "
        "Báo cáo nên có accuracy, macro-F1, số tham số, thời gian/epoch, FPS inference và VRAM."
    )
    doc.add_heading("Khác biệt cốt lõi giữa AlexNet và Transformer", level=2)
    add_table(
        doc,
        ["AlexNet/CNN", "Custom Vision Transformer"],
        [
            ("Kernel quét cục bộ trên ảnh", "Chia ảnh thành token patch"),
            ("Quan hệ xa hình thành dần qua nhiều convolution", "Self-Attention cho token trao đổi trực tiếp"),
            ("Pooling giảm kích thước không gian", "CLS tổng hợp thông tin toàn ảnh"),
            ("Vị trí được ngầm giữ bởi lưới convolution", "Cần Positional Embedding"),
            ("Dễ train hơn từ đầu trên dữ liệu nhỏ", "Thường cần augmentation và regularization tốt"),
        ],
        widths=[8, 8],
    )

    doc.add_heading("5. Khả năng triển khai Web, Mobile và WinForms", level=1)
    doc.add_picture(str(deployment), width=Inches(6.7))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(
        "Không cần chuyển sang TensorFlow chỉ vì muốn làm ứng dụng. Có thể train bằng PyTorch, export ONNX và dùng cùng model cho nhiều nền tảng. "
        "Điều bắt buộc là preprocessing trong app phải giống data.py: RGB, resize 112×112, chuyển tensor và normalize mean/std 0.5."
    )
    add_table(
        doc,
        ["Nền tảng", "Hướng đề xuất"],
        [
            ("Web App", "Ưu tiên FastAPI/Streamlit chạy inference phía server; sau đó cân nhắc ONNX Runtime Web."),
            ("Mobile", "ONNX Runtime Mobile hoặc ExecuTorch; benchmark latency và dung lượng model trên thiết bị thật."),
            ("WinForms", "Export ONNX và dùng Microsoft.ML.OnnxRuntime trong C#."),
        ],
        widths=[4, 12],
    )

    doc.add_heading("6. Kết luận và bước tiếp theo", level=1)
    doc.add_paragraph(
        "DataLoader và forward pass đã được kiểm tra bằng dữ liệu thật. Model hiện tạo đúng logits [16,105] và có 1.366.185 tham số. "
        "Tuy nhiên, chưa có kết luận về accuracy vì model chưa được huấn luyện."
    )
    steps = [
        "Viết train.py: device, model, CrossEntropyLoss, AdamW.",
        "Viết một epoch train: forward → loss → backward → optimizer.step.",
        "Viết validation không cập nhật trọng số.",
        "Lưu checkpoint tốt nhất và Early Stopping patience=7.",
        "Vẽ loss/accuracy, confusion matrix và đánh giá test set.",
        "Chuyển classifier sang face embedding/ArcFace để hỗ trợ Unknown.",
        "Tích hợp Face Detection nhiều khuôn mặt và đánh giá khuôn mặt nhỏ.",
        "Export ONNX và triển khai ứng dụng.",
    ]
    for index, step in enumerate(steps, start=1):
        doc.add_paragraph(f"{index}. {step}")

    doc.add_heading("6.1. Cơ sở học thuật và tài liệu gốc", level=2)
    doc.add_paragraph(
        "SIC-ViT-4 không phải là tên một model đã công bố và không nên ghi trong báo cáo rằng đây là kiến trúc mới. "
        "Cách gọi đúng: custom implementation / cấu hình thực nghiệm lấy cảm hứng từ các công trình sau:"
    )
    references = [
        "Vaswani et al., Attention Is All You Need, NeurIPS 2017, arXiv:1706.03762 – nguồn gốc Transformer và Self-Attention.",
        "Dosovitskiy et al., An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, ICLR 2021, arXiv:2010.11929 – Vision Transformer (ViT) gốc.",
        "Touvron et al., Training data-efficient image transformers & distillation through attention, ICML 2021, arXiv:2012.12877 – DeiT và recipe train hiệu quả dữ liệu.",
        "Liu et al., Swin Transformer: Hierarchical Vision Transformer using Shifted Windows, ICCV 2021, arXiv:2103.14030 – Transformer phân cấp với window attention.",
        "Mehta and Rastegari, MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer, ICLR 2022, arXiv:2110.02178 – hướng Transformer nhẹ cho mobile.",
        "Deng et al., ArcFace: Additive Angular Margin Loss for Deep Face Recognition, CVPR 2019, arXiv:1801.07698 – cơ sở cho phase face embedding/face recognition thật sự.",
    ]
    for reference in references:
        doc.add_paragraph(reference, style="List Bullet")
    doc.add_paragraph(
        "Lưu ý: model hiện tại dùng classification head và CrossEntropyLoss để học pipeline. Khi chuyển sang nhận diện người lạ và nhiều khuôn mặt, backbone Transformer sẽ cần được nối với embedding head và ArcFace/CosFace; đó mới là phase face recognition hoàn chỉnh."
    )

    doc.add_heading("7. Từ khóa cần nhớ", level=1)
    add_table(
        doc,
        ["Từ khóa", "Định nghĩa ngắn"],
        [
            ("Tensor", "Khối số nhiều chiều dùng cho tính toán neural network."),
            ("Weight", "Tham số model tự điều chỉnh khi training."),
            ("Logits", "Điểm thô trước Softmax."),
            ("Epoch", "Một lượt model đi qua toàn bộ train set."),
            ("Batch", "Một nhóm ảnh được xử lý và cập nhật trọng số cùng lượt."),
            ("Overfitting", "Train tốt nhưng validation/test kém."),
            ("Patch", "Một vùng ảnh nhỏ được xem như token."),
            ("Embed dimension", "Số đặc trưng dùng để biểu diễn mỗi token."),
            ("Attention", "Cơ chế token lấy thông tin từ token khác theo mức liên quan."),
            ("Residual", "Cộng đầu vào cũ với thông tin mới của layer."),
        ],
        widths=[5, 11],
    )

    doc.add_page_break()
    doc.add_heading("8. Giải thích chi tiết các thành phần bên trong Transformer", level=1)
    doc.add_paragraph(
        "Phần này giải thích từng thành phần theo đúng thứ tự chạy trong "
        "TransformerEncoderBlock. Tensor đi vào block có dạng [B,T,E]. Với cấu hình "
        "SIC-ViT-4: B là batch size, T=50 token (1 CLS + 49 patch), E=192 đặc trưng."
    )
    add_code(
        doc,
        "x [B,T,E]\n"
        "x = x + Attention(LayerNorm(x))\n"
        "x = x + FeedForward(LayerNorm(x))\n"
        "output [B,T,E]",
    )

    doc.add_heading("8.1. LayerNorm là gì?", level=2)
    doc.add_paragraph(
        "LayerNorm (Layer Normalization) chuẩn hóa các giá trị đặc trưng bên trong từng "
        "token. Nó không trộn các ảnh với nhau và cũng không trộn các token với nhau. "
        "Với tensor [B,T,E], LayerNorm tính trung bình và độ lệch chuẩn dọc theo chiều "
        "cuối E. Vì E=192, mỗi token được chuẩn hóa riêng trên 192 giá trị của chính nó."
    )
    add_code(
        doc,
        "Input:  x [16,50,192]\n"
        "Có 16 x 50 = 800 token\n"
        "Mỗi token gồm 192 số và được LayerNorm riêng\n"
        "Output: [16,50,192]  (shape không đổi)",
    )
    doc.add_paragraph(
        "Cách tính trực giác: lấy một token, tính mean, trừ mean để đưa dữ liệu về quanh "
        "0, rồi chia cho độ lệch chuẩn để các giá trị có độ lớn ổn định. Sau đó LayerNorm "
        "dùng hai tham số học được gamma và beta để model vẫn có thể điều chỉnh lại tỷ lệ "
        "và độ dịch chuyển phù hợp."
    )
    add_code(
        doc,
        "mean = trung bình 192 đặc trưng\n"
        "variance = trung bình bình phương khoảng cách tới mean\n"
        "normalized = (x - mean) / sqrt(variance + epsilon)\n"
        "output = gamma * normalized + beta",
    )
    doc.add_paragraph(
        "Ví dụ rút gọn với token [2,4,6]: mean=4, sau khi trừ mean thành [-2,0,2]. "
        "LayerNorm tiếp tục chia cho độ lệch chuẩn. Mục đích không phải làm mất thông tin, "
        "mà đưa các đặc trưng về thang đo ổn định trước khi đi vào Attention hoặc MLP."
    )
    add_table(
        doc,
        ["Điểm cần nhớ", "LayerNorm trong model này"],
        [
            ("Chuẩn hóa theo chiều nào?", "Chiều đặc trưng E=192 của từng token."),
            ("Có đổi shape không?", "Không: [B,T,192] vẫn là [B,T,192]."),
            ("Có tham số học không?", "Có: gamma và beta, mỗi tham số có 192 giá trị."),
            ("Tại sao cần?", "Giữ độ lớn dữ liệu ổn định, giúp gradient và quá trình train ổn định hơn."),
            ("Khác BatchNorm?", "LayerNorm không phụ thuộc các ảnh khác trong batch; phù hợp chuỗi token và batch nhỏ."),
        ],
        widths=[5, 11],
    )

    doc.add_heading("8.2. Pre-Norm và hai LayerNorm", level=2)
    doc.add_paragraph(
        "Block có hai LayerNorm khác nhau. norm_attention chuẩn hóa trước Attention; "
        "norm_mlp chuẩn hóa trước FeedForward. Chúng có cùng cách hoạt động nhưng sở hữu "
        "gamma và beta riêng, vì dữ liệu ở hai vị trí đã khác nhau. Cách đặt LayerNorm "
        "trước module được gọi là Pre-Norm."
    )
    add_code(
        doc,
        "attention_input = norm_attention(x)\n"
        "attention_output = self_attention(attention_input)\n"
        "x = x + attention_output\n\n"
        "mlp_input = norm_mlp(x)\n"
        "mlp_output = feed_forward(mlp_input)\n"
        "x = x + mlp_output",
    )

    doc.add_heading("8.3. Self-Attention hoạt động như thế nào?", level=2)
    doc.add_paragraph(
        "Self-Attention cho phép mỗi token lấy thông tin từ tất cả token còn lại. Một patch "
        "chứa mắt có thể chú ý tới patch chứa mũi, miệng hoặc đường nét khuôn mặt. CLS token "
        "cũng chú ý tới 49 patch để dần tổng hợp thông tin của toàn ảnh."
    )
    add_table(
        doc,
        ["Thành phần", "Câu hỏi trực giác", "Vai trò"],
        [
            ("Query (Q)", "Token này đang tìm thông tin gì?", "Đại diện nhu cầu tìm kiếm của token hiện tại."),
            ("Key (K)", "Token kia phù hợp đến mức nào?", "Dùng để tính độ liên quan với Query."),
            ("Value (V)", "Nếu phù hợp thì lấy nội dung gì?", "Thông tin được cộng có trọng số vào kết quả."),
        ],
        widths=[3, 5, 8],
    )
    add_code(
        doc,
        "scores = Q @ K^T / sqrt(head_dim)\n"
        "attention_weights = softmax(scores)\n"
        "output = attention_weights @ V",
    )
    doc.add_paragraph(
        "Softmax biến các điểm liên quan thành trọng số có tổng bằng 1. Token liên quan hơn "
        "nhận trọng số lớn hơn. Phép chia cho căn bậc hai của head_dim giữ scores không quá "
        "lớn, tránh Softmax bị bão hòa và giúp train ổn định."
    )

    doc.add_heading("8.4. Multi-Head Attention là gì?", level=2)
    doc.add_paragraph(
        "Thay vì chỉ thực hiện một Attention trên toàn bộ 192 chiều, model chia thành 3 head. "
        "Mỗi head xử lý 64 chiều vì 192/3=64. Các head có trọng số riêng nên có thể học "
        "những kiểu quan hệ khác nhau, chẳng hạn một head chú ý hình dạng mắt, một head chú ý "
        "tỷ lệ giữa các bộ phận và một head chú ý đường nét tổng thể. Đây chỉ là cách hiểu trực "
        "giác; model tự quyết định mỗi head thực sự học gì."
    )
    add_code(
        doc,
        "Input [B,50,192]\n"
        "3 heads x 64 chiều\n"
        "Attention chạy song song trên 3 head\n"
        "Ghép kết quả: [B,50,192]\n"
        "Output projection: [B,50,192]",
    )

    doc.add_heading("8.5. Residual connection là gì?", level=2)
    doc.add_paragraph(
        "Residual connection cộng đầu vào cũ với thông tin mới do Attention hoặc FeedForward "
        "tạo ra. Module không phải xây lại toàn bộ biểu diễn từ đầu; nó chỉ cần học phần thông "
        "tin cần bổ sung. Đường cộng trực tiếp cũng giúp gradient truyền qua nhiều block dễ hơn."
    )
    add_code(doc, "x_new = x_old + new_information")
    doc.add_paragraph(
        "Hai tensor phải cùng shape để cộng theo từng vị trí. Đây là lý do Attention và "
        "FeedForward đều đưa output trở lại [B,T,E]."
    )

    doc.add_heading("8.6. FeedForward xử lý gì?", level=2)
    doc.add_paragraph(
        "Attention trao đổi thông tin giữa các token. FeedForward lại xử lý từng token riêng, "
        "nhưng dùng cùng một bộ trọng số cho mọi token. Trong model hiện tại, mỗi vector 192 "
        "chiều được mở rộng lên 384 chiều, đi qua GELU và Dropout, rồi thu về 192 chiều."
    )
    add_code(
        doc,
        "[B,T,192]\n"
        "-> Linear(192,384)\n"
        "-> GELU\n"
        "-> Dropout\n"
        "-> Linear(384,192)\n"
        "-> Dropout\n"
        "-> [B,T,192]",
    )
    add_table(
        doc,
        ["Thành phần", "Ý nghĩa"],
        [
            ("Linear 192->384", "Mở rộng không gian để học tổ hợp đặc trưng phong phú hơn."),
            ("GELU", "Thêm tính phi tuyến; nếu chỉ có Linear liên tiếp thì khả năng biểu diễn bị giới hạn."),
            ("Dropout", "Khi train, tắt ngẫu nhiên một phần giá trị để giảm phụ thuộc và hạn chế overfitting."),
            ("Linear 384->192", "Đưa tensor về embed_dim để có thể cộng residual với đầu vào."),
        ],
        widths=[5, 11],
    )

    doc.add_heading("8.7. Toàn bộ luồng của một block", level=2)
    add_table(
        doc,
        ["Bước", "Phép xử lý", "Shape"],
        [
            ("1", "Nhận token x", "[B,50,192]"),
            ("2", "norm_attention(x)", "[B,50,192]"),
            ("3", "Multi-Head Self-Attention", "[B,50,192]"),
            ("4", "Cộng residual với x cũ", "[B,50,192]"),
            ("5", "norm_mlp(x)", "[B,50,192]"),
            ("6", "FeedForward 192->384->192", "[B,50,192]"),
            ("7", "Cộng residual lần hai", "[B,50,192]"),
        ],
        widths=[2, 9, 5],
    )
    doc.add_paragraph(
        "SIC-ViT-4 lặp lại luồng trên bốn lần; cấu hình depth=12 lặp lại mười hai lần. "
        "Depth đếm số TransformerEncoderBlock, không phải tổng số Linear, LayerNorm hay "
        "Attention nhỏ nằm bên trong các block."
    )

    doc.save(OUTPUT_PATH)
    return OUTPUT_PATH


if __name__ == "__main__":
    print(build_document())
