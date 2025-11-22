import streamlit as st

st.set_page_config(page_title="DigiarSip", layout="wide")

import os
from pathlib import Path

# cache folder aman cloud
Path("/tmp/docling_artifacts").mkdir(parents=True, exist_ok=True)
os.environ["DOCLING_ARTIFACTS_PATH"] = "/tmp/docling_artifacts"

import os
import gc
import io
import cv2
import base64
import pathlib
import zipfile
import tempfile
import numpy as np
import mimetypes
import pytesseract 
import re
import shutil
import PIL.ImageEnhance as ImageEnhance
import html
from PIL import ImageFilter
from PyPDF2 import PdfReader
from skimage.filters import threshold_local
from pathlib import Path
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import gdown
import torch
import torchvision.transforms as torchvision_T
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_mobilenet_v3_large
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, ImageFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from modelscope import snapshot_download
from docling.datamodel.pipeline_options import RapidOcrOptions

# ==========================
# OCR binary path (Windows)
# ==========================
# ==========================
# OCR binary path (Windows only)
# ==========================
import platform
if platform.system().lower() == "windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# ==========================
# Model loading
# ==========================
# @st.cache(allow_output_mutation=True)
# ========= Google Drive Model IDs =========
MBV3_ID = "1kJNrlX5iWlrNF3yA88FmhHm9n4tBElOg"
R50_ID  = "13JcqlolBrUypnt2gzDikj9BfV2BqHM-j"

MBV3_URL = f"https://drive.google.com/uc?id={MBV3_ID}"
R50_URL  = f"https://drive.google.com/uc?id={R50_ID}"

# ========= Path aman untuk Streamlit Cloud =========
MODEL_DIR = Path("/tmp/digiarsip_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MBV3_PATH = MODEL_DIR / "model_mbv3_iou_mix_2C049.pth"
R50_PATH  = MODEL_DIR / "model_r50_iou_mix_2C020.pth"


def download_if_missing(path: Path, url: str):
    """Download model dari Google Drive jika belum ada."""
    if not path.exists():
        print(f"[INFO] Downloading model to: {path}")
        gdown.download(url, str(path), quiet=False)


# ============= MODEL LOADING FIXED VERSION =============
def load_model(num_classes=2, model_name="mbv3", device=torch.device("cpu")):
    # Tentukan model dan path checkpoint
    if model_name == "mbv3":
        model = deeplabv3_mobilenet_v3_large(num_classes=num_classes, aux_loss=True)
        checkpoint_path = MBV3_PATH
        checkpoint_url  = MBV3_URL
    else:
        model = deeplabv3_resnet50(num_classes=num_classes, aux_loss=True)
        checkpoint_path = R50_PATH
        checkpoint_url  = R50_URL

    # Download model jika belum ada di /tmp
    download_if_missing(checkpoint_path, checkpoint_url)

    # Load model
    model.to(device)
    checkpoints = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoints, strict=False)
    model.eval()

    # warm-up (skip di Streamlit Cloud untuk percepat startup)
    try:
        if os.environ.get("STREAMLIT_CLOUD") != "1":
            _ = model(torch.randn((1, 3, 384, 384)))
    except Exception:
        pass

    return model

@st.cache_resource
def get_model_cached(model_name="mbv3"):
    # load_model kamu tetap dipakai, tapi sekarang dicache
    return load_model(model_name=model_name)
def image_preprocess_transforms(mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)):
    common_transforms = torchvision_T.Compose(
        [
            torchvision_T.ToTensor(),
            torchvision_T.Normalize(mean, std),
        ]
    )
    return common_transforms

# ==========================
# Geometry helpers (perspective)
# ==========================

def order_points(pts):
    """Rearrange coordinates to order: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect.astype("int").tolist()


def find_dest(pts):
    (tl, tr, br, bl) = pts
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
    return order_points(destination_corners)

# ==========================
# Segmentation-based scan (your existing pipeline)
# ==========================

def scan(image_true=None, trained_model=None, image_size=384, BUFFER=10):
    global preprocess_transforms

    IMAGE_SIZE = image_size
    half = IMAGE_SIZE // 2

    imH, imW, C = image_true.shape

    image_model = cv2.resize(image_true, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

    scale_x = imW / IMAGE_SIZE
    scale_y = imH / IMAGE_SIZE

    image_model = preprocess_transforms(image_model)
    image_model = torch.unsqueeze(image_model, dim=0)

    with torch.no_grad():
        out = trained_model(image_model)["out"].cpu()

    del image_model
    gc.collect()

    out = torch.argmax(out, dim=1, keepdims=True).permute(0, 2, 3, 1)[0].numpy().squeeze().astype(np.int32)
    r_H, r_W = out.shape

    _out_extended = np.zeros((IMAGE_SIZE + r_H, IMAGE_SIZE + r_W), dtype=out.dtype)
    _out_extended[half : half + IMAGE_SIZE, half : half + IMAGE_SIZE] = out * 255
    out = _out_extended.copy()

    del _out_extended
    gc.collect()

    canny = cv2.Canny(out.astype(np.uint8), 225, 255)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    page = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    epsilon = 0.02 * cv2.arcLength(page, True)
    corners = cv2.approxPolyDP(page, epsilon, True)

    corners = np.concatenate(corners).astype(np.float32)

    corners[:, 0] -= half
    corners[:, 1] -= half

    corners[:, 0] *= scale_x
    corners[:, 1] *= scale_y

    if not (np.all(corners.min(axis=0) >= (0, 0)) and np.all(corners.max(axis=0) <= (imW, imH))):

        left_pad, top_pad, right_pad, bottom_pad = 0, 0, 0, 0

        rect = cv2.minAreaRect(corners.reshape((-1, 1, 2)))
        box = cv2.boxPoints(rect)
        box_corners = np.int32(box)

        box_x_min = np.min(box_corners[:, 0])
        box_x_max = np.max(box_corners[:, 0])
        box_y_min = np.min(box_corners[:, 1])
        box_y_max = np.max(box_corners[:, 1])

        if box_x_min <= 0:
            left_pad = abs(box_x_min) + BUFFER
        if box_x_max >= imW:
            right_pad = (box_x_max - imW) + BUFFER
        if box_y_min <= 0:
            top_pad = abs(box_y_min) + BUFFER
        if box_y_max >= imH:
            bottom_pad = (box_y_max - imH) + BUFFER

        image_extended = np.zeros((top_pad + bottom_pad + imH, left_pad + right_pad + imW, C), dtype=image_true.dtype)
        image_extended[top_pad : top_pad + imH, left_pad : left_pad + imW, :] = image_true
        image_extended = image_extended.astype(np.float32)

        box_corners[:, 0] += left_pad
        box_corners[:, 1] += top_pad

        corners = box_corners
        image_true = image_extended

    corners = sorted(corners.tolist())
    corners = order_points(corners)
    destination_corners = find_dest(corners)
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))

    final = cv2.warpPerspective(image_true, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LANCZOS4)
    final = np.clip(final, a_min=0, a_max=255)
    final = final.astype(np.uint8)

    return final
# ==========================
# Docling Setup
# ==========================
# docling_available = False
# doc_converter = None
# try:
#     from docling.datamodel.base_models import InputFormat
#     from docling.document_converter import DocumentConverter, ImageFormatOption
#     from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode


#     pipeline_options = PdfPipelineOptions(do_table_structure=True)
#     pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE


#     doc_converter = DocumentConverter(
#     format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options)},
#     allowed_formats=[InputFormat.IMAGE],
#     )
#     docling_available = True
# except Exception:
#     docling_available = False
# ==========================
# CamScanner-like enhancement (merged)
# ==========================

def gray_world_white_balance(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    means = img.reshape(-1,3).mean(axis=0)
    overall = means.mean()
    scales = overall / (means + 1e-6)
    out = np.clip(img * scales, 0, 255).astype(np.uint8)
    return out


def illumination_correction(gray: np.ndarray) -> np.ndarray:
    """Remove gradual lighting using large-kernel blur division."""
    blur = cv2.GaussianBlur(gray, (0,0), sigmaX=13)
    corrected = cv2.divide(gray, blur, scale=255)
    return corrected


def enhance_doc(bgr: np.ndarray, grayscale: bool=False, clip_limit: float=3.0, tile: int=8, sharpen_amount: float=1.0) -> np.ndarray:
    # white balance
    bgr = gray_world_white_balance(bgr)

    # convert to LAB for CLAHE on L
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # illumination correction on L
    L = illumination_correction(L)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    L = clahe.apply(L)

    lab = cv2.merge([L, A, B])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # unsharp mask
    if sharpen_amount > 0:
        blur = cv2.GaussianBlur(bgr, (0,0), sigmaX=1.0)
        bgr = cv2.addWeighted(bgr, 1 + 0.6*sharpen_amount, blur, -0.6*sharpen_amount, 0)

    if grayscale:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr


grayscale = False
# ==========================
# OCR helpers
# ==========================
def get_rapidocr_options():
    """
    Download RapidOCR models sekali, lalu kembalikan RapidOcrOptions
    dengan path model ONNX yang lengkap.
    """
    # cache path aman cloud
    cache_dir = Path("/tmp/rapidocr_models")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # download RapidOCR repo (sekali per container)
    download_path = snapshot_download(
        repo_id="RapidAI/RapidOCR",
        cache_dir=str(cache_dir)
    )

    # pakai PP-OCRv4 ONNX (sesuai error kamu)
    det_model_path = os.path.join(
        download_path, "onnx", "PP-OCRv4", "det", "ch_PP-OCRv4_det_infer.onnx"
    )
    rec_model_path = os.path.join(
        download_path, "onnx", "PP-OCRv4", "rec", "ch_PP-OCRv4_rec_infer.onnx"
    )
    cls_model_path = os.path.join(
        download_path, "onnx", "PP-OCRv4", "cls", "ch_ppocr_mobile_v2.0_cls_infer.onnx"
    )

    return RapidOcrOptions(
        det_model_path=det_model_path,
        rec_model_path=rec_model_path,
        cls_model_path=cls_model_path,
    )
from rapidocr_onnxruntime import RapidOCR  # <- dari rapidocr-onnxruntime

@st.cache_resource
def get_rapidocr_engine():
    cache_dir = Path("/tmp/rapidocr_models")
    cache_dir.mkdir(parents=True, exist_ok=True)

    download_path = snapshot_download(
        repo_id="RapidAI/RapidOCR",
        cache_dir=str(cache_dir)
    )

    det_model_path = os.path.join(
        download_path, "onnx", "PP-OCRv4", "det", "ch_PP-OCRv4_det_infer.onnx"
    )
    rec_model_path = os.path.join(
        download_path, "onnx", "PP-OCRv4", "rec", "ch_PP-OCRv4_rec_infer.onnx"
    )
    cls_model_path = os.path.join(
        download_path, "onnx", "PP-OCRv4", "cls", "ch_ppocr_mobile_v2.0_cls_infer.onnx"
    )

    engine = RapidOCR(
        det_model_path=det_model_path,
        rec_model_path=rec_model_path,
        cls_model_path=cls_model_path,
        use_angle_cls=True,   # <-- tambah ini biar tahan miring
    )
    return engine


def preprocess_for_ocr(bgr: np.ndarray) -> np.ndarray:
    """
    Preprocess ringan tapi stabil untuk OCR di cloud:
    - grayscale
    - denoise halus
    - adaptive threshold (biar teks tegas)
    - sedikit sharpen
    Output: grayscale/binary image
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # denoise ringan
    gray = cv2.fastNlMeansDenoising(gray, None, 7, 7, 21)

    # adaptive threshold lokal (lebih stabil di scan gelap/terang)
    t = threshold_local(gray, block_size=35, offset=10, method="gaussian")
    bw = (gray > t).astype("uint8") * 255

    # sharpen tipis
    blur = cv2.GaussianBlur(bw, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(bw, 1.4, blur, -0.4, 0)

    return sharp

def extract_text_from_image(path: Path) -> str:
    """
    Pipeline OCR stabil untuk Streamlit Cloud:
    1) baca image
    2) preprocess OCR
    3) RapidOCR ONNX
    4) fallback Tesseract kalau kosong
    5) normalisasi output (spasi, URL, kata menempel)
    """
    ocr_engine = get_rapidocr_engine()

    img = cv2.imread(str(path))
    if img is None:
        return ""

    # preprocess khusus OCR
    ocr_img = preprocess_for_ocr(img)

    try:
        result, _ = ocr_engine(ocr_img)
        texts = []
        if result:
            for line in result:
                if len(line) >= 2:
                    texts.append(line[1])

        text = " ".join(texts).strip()

    except Exception as e:
        print(f"[RapidOCR failed] {e}")
        text = ""

    # fallback tesseract kalau RapidOCR kosong
    if len(text) < 5:
        try:
            text = pytesseract.image_to_string(ocr_img, lang="ind+eng").strip()
        except Exception:
            text = ""

    # normalisasi akhir biar konsisten format metadata
    text = html.unescape(text or "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)           # CamelCase nempel
    text = re.sub(r"http\s*:\s*/\s*/", "http://", text, flags=re.I)
    text = re.sub(r"https\s*:\s*/\s*/", "https://", text, flags=re.I)
    text = re.sub(r"\s*/\s*", "/", text)                       # rapikan slash
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\n{2,}", "\n", text)

    return text.strip()



    # img = cv2.imread(str(path))
    # if img is None:
    #     return ""
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # text = pytesseract.image_to_string(gray, lang="ind+eng")
    # return text.strip()
    # if not docling_available or doc_converter is None:
    #     return ""
    # try:
    #     # Convert to grayscale before sending to Docling
    #     img = cv2.imread(str(path))
    #     if img is None:
    #         return ""
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     tmp_path = path.parent / f"{path.stem}_gray.png"
    #     cv2.imwrite(str(tmp_path), gray)


    #     result = doc_converter.convert(str(tmp_path))
    #     if hasattr(result, "document") and result.document is not None:
    #         doc = result.document
    #         if hasattr(doc, "export_to_text"):
    #             return (doc.export_to_text() or "").strip()
    #         if hasattr(doc, "to_text"):
    #             return (doc.to_text() or "").strip()
    #         if hasattr(doc, "pages"):
    #             return "\n".join((getattr(p, "text", "") or "").strip() for p in doc.pages)
    #     if hasattr(result, "export_to_text"):
    #         return (result.export_to_text() or "").strip()
    # except Exception:
    #     return ""
    # return ""

# def extract_text_from_image(path: Path) -> str:
#     if not docling_available or doc_converter is None:
#         return ""
#     try:
#         # Baca & ubah gambar ke grayscale
#         img = cv2.imread(str(path))
#         if img is None:
#             return ""
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         tmp_path = path.parent / f"{path.stem}_gray.png"
#         cv2.imwrite(str(tmp_path), gray)

#         # Konversi dokumen menggunakan Docling
#         result = doc_converter.convert(str(tmp_path))
#         text = ""

#         # Ambil teks secepat mungkin (tidak perlu semua atribut dicek panjang)
#         if hasattr(result, "document") and result.document:
#             doc = result.document
#             if hasattr(doc, "export_to_text"):
#                 text = doc.export_to_text() or ""
#             elif hasattr(doc, "to_text"):
#                 text = doc.to_text() or ""
#             elif hasattr(doc, "pages"):
#                 text = "\n".join((getattr(p, "text", "") or "").strip() for p in doc.pages)
#         elif hasattr(result, "export_to_text"):
#             text = result.export_to_text() or ""

#         # Batasi output ke 50 kata untuk efisiensi
#         words = text.strip().split()
#         limited_text = " ".join(words[:20])
#         return limited_text

#     except Exception:
#         return ""


# def extract_text_from_pdf(pdf_path: Path) -> str:
#     # try:
#     #     reader = PdfReader(str(pdf_path))
#     #     text = ""
#     #     for page in reader.pages[:2]:
#     #         text += page.extract_text() or ""
#     #     if text.strip():
#     #         return text.strip()
#     # except Exception:
#     #     pass
#     # try:
#     #     import pdf2image
#     #     pages = pdf2image.convert_from_path(str(pdf_path), first_page=1, last_page=1)
#     #     if pages:
#     #         tmp = np.array(pages[0])
#     #         text = pytesseract.image_to_string(tmp, lang="ind+eng")
#     #         return text.strip()
#     # except Exception:
#     #     pass
#     # return ""
#     if not docling_available or doc_converter is None:
#         return ""
#     try:
#         result = doc_converter.convert(str(pdf_path))
#         if hasattr(result, "document") and result.document is not None:
#             doc = result.document
#             if hasattr(doc, "export_to_text"):
#                 return (doc.export_to_text() or "").strip()
#             if hasattr(doc, "to_text"):
#                 return (doc.to_text() or "").strip()
#             if hasattr(doc, "pages"):
#                 return "\n".join((getattr(p, "text", "") or "").strip() for p in doc.pages)
#         if hasattr(result, "export_to_text"):
#             return (result.export_to_text() or "").strip()
#     except Exception:
#         return ""
#     return ""
def normalize_field_linebreaks(s: str) -> str:
    # Satukan label yang jatuh ke baris berikutnya
    s = re.sub(r"(?:Nomor|No\.?|Perihal|Hal|Tentang)\s*[:\-–]?\s*\n+\s*", 
               lambda m: m.group(0).split()[0].rstrip(":") + ": ", 
               s, flags=re.IGNORECASE)
    # Rapikan spasi berlebih
    s = re.sub(r"[ \t]+", " ", s)
    return s
HEADER_PATTERNS = [
    r"\bPEMERINTAH\b", r"\bKABUPATEN\b", r"\bKOTA\b", r"\bDINAS\b", r"\bBADAN\b",
    r"\bSEKRETARIAT\b", r"\bBAGIAN\b", r"\bJL\.?\b", r"\bJALAN\b",
    r"\bTELP\b|\bTELEPON\b|\bFAX\b|\bEMAIL\b|\bE-MAIL\b|\bWEBSITE\b|\bSITUS\b",
    r"\bKODE\s*POS\b", r"\bPROVINSI\b"
]
HEADER_RE = re.compile("|".join(HEADER_PATTERNS), re.IGNORECASE)
# --- helper: ambil jendela tepat setelah label "Nomor/No." ---
NOMOR_LABEL = re.compile(r"(?:^|\n)\s*(?:Nomor|No\.?)\s*[:\-–]?\s*", re.IGNORECASE)
NEXT_FIELD_LABEL = re.compile(
    r"(Sifat|Lampiran|Perihal|Hal|Tentang|Kepada(?:\s*Yth\.?)?|Yth\.?|Tembusan)\s*[:\-–]?",
    re.IGNORECASE,
)

def extract_nomor_window(text: str, span: int = 360) -> str:
    t = text or ""
    m = NOMOR_LABEL.search(t)
    if not m:
        return t
    tail = t[m.end(): m.end()+span]

    # ⬇️ potong jika ada label berikutnya DI MANA PUN (tidak pakai ^)
    n = NEXT_FIELD_LABEL.search(tail)
    if n:
        tail = tail[:n.start()]

    # rapikan spasi berlebih
    tail = re.sub(r"[ \t]+", " ", tail).strip()
    return tail


# --- helper: normalisasi karakter mirip digit di segmen numerik ---
def fix_digit_lookalikes(seg: str) -> str:
    # hanya terapkan kalau segmen terlihat numerik (digit/titik/dash mayoritas)
    if not re.fullmatch(r"[A-Za-z0-9.\-]+", seg or ""):
        return seg
    s = seg
    # konversi yang umum di OCR:
    s = re.sub(r"[oO]", "0", s)
    s = re.sub(r"[lI!]", "1", s)
    # 'b' kadang muncul di antara digit → 6 (heuristik aman)
    s = re.sub(r"(?<=\d)[bB](?=\d)", "6", s)
    # 'S' kadang untuk 5 jika dikelilingi digit
    s = re.sub(r"(?<=\d)[sS](?=\d)", "5", s)
    # kompres pemisah berulang
    s = re.sub(r"([.\-])\1+", r"\1", s)
    return s

LINE_LABEL_FIXES = [
    (r"(?im)^(Nomor)\s*:\s*:", r"\1: "),
    (r"(?im)^(Perihal|Hal|Tentang)\s*:\s*:", r"\1: "),
]

def normalize_for_fields(text: str) -> str:
    t = (text or "")
    t = normalize_text(t)                        # buang spasi/blank line berlebih
    t = normalize_field_linebreaks(t)            # “Nomor:\n 445/..” → “Nomor: 445/..”
    for patt, repl in LINE_LABEL_FIXES:          # “Nomor: : …” → “Nomor: …”
        t = re.sub(patt, repl, t)
    return t
# --- helper: roman numeral ---
# --- Roman helper (tetap jaga 'b') ---
ROMAN_MONTHS = {"I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII"}
def _roman_like(tok: str) -> bool:
    return bool(tok) and re.fullmatch(r"[IVXLCDM1l]+", tok.upper())

def repair_roman(tok: str) -> str:
    u = tok.upper().replace("L", "I").replace("1", "I")
    u = re.sub(r"I{5,}", "III", u)
    return u if u in ROMAN_MONTHS else tok

def _collapse_seg_spaces(s: str) -> str:
    # runtuhkan spasi di dalam segmen antar-slash: '/2b 66/' → '/2b66/'
    return re.sub(r"(?<=/)[A-Za-z0-9 ]+(?=/|$)", lambda m: m.group(0).replace(" ", ""), s)

def _tidy_separators(s: str) -> str:
    s = re.sub(r"\s*/\s*", "/", s)
    s = re.sub(r"\s*\.\s*", ".", s)
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"([/.\-])\1+", r"\1", s)
    return s

def find_nomor_surat(text: str, image_path: Path = None) -> str:
    if not text:
        return ""
    t = normalize_for_fields(text)

    # 1) tangkap baris "Nomor: ..." atau baris setelahnya (kalau patah)
    m = re.search(r"(?im)^(?:Nomor|No\.)\s*:\s*(.+)$", t)
    window = ""
    if m:
        line = m.group(1).strip()
        # ambil juga baris setelahnya kalau baris pertama pendek
        lines = t.splitlines()
        idx = next((i for i, ln in enumerate(lines) if re.search(r"(?im)^(?:Nomor|No\.)\s*:\s*", ln)), None)
        nxt = (lines[idx+1].strip() if idx is not None and idx+1 < len(lines) else "")
        window = (line + " " + nxt).strip()
    else:
        # fallback: gunakan seluruh teks (lebih lemah)
        window = t

    # 2) cari kandidat di window (izinkan spasi di segmen)
    token = r"[A-Za-z0-9]+(?:[ .][A-Za-z0-9]+)*"
    pat   = rf"({token}(?:\s*/\s*{token}){{2,12}}(?:\s*/\s*20\d{{2}})?)"
    cands = re.findall(pat, window)

    if not cands:
        return ""

    def normalize_candidate(s: str) -> str:
        s = _tidy_separators(s.strip())
        s = _collapse_seg_spaces(s)
        # perbaiki roman tapi jangan ubah 'b'
        segs = []
        for seg in s.split("/"):
            parts = seg.split(".")
            fixed = []
            for p in parts:
                fixed.append(repair_roman(p) if _roman_like(p) else p)
            segs.append(".".join(fixed))
        s = "/".join(segs)
        s = re.sub(r"[^\w./-]+", "", s)
        return _tidy_separators(s)

    scored = []
    for c in cands:
        n = normalize_candidate(c)
        # 2a) jika ada tahun, stop setelah '/20xx'
        m_year = re.search(r"/20\d{2}(?!\d)", n)
        if m_year:
            n = n[:m_year.end()]

        # 2b) fallback: buang trailing label bila masih ada, meski tanpa newline
        m_lbl = re.search(NEXT_FIELD_LABEL, n)
        if m_lbl:
            n = n[:m_lbl.start()].rstrip(" /.-")

        # tolak yang terlalu generik (tanpa slash atau segmen < 3)
        if n.count("/") < 2:
            continue
        # nilai tinggi: banyak slash, ada titik di segmen awal (kode 500.12.15.1), panjang
        score = 5*n.count("/") + 3*n.split("/")[0].count(".") + len(n)/30.0
        scored.append((score, n))

    if not scored:
        return ""

    scored.sort(reverse=True)
    return scored[0][1]


# =======================
# Helper: normalisasi tanggal & angka
# =======================
ID_MONTHS = {
    "januari":"01","februari":"02","maret":"03","april":"04","mei":"05","juni":"06",
    "juli":"07","agustus":"08","september":"09","oktober":"10","november":"11","desember":"12"
}
def parse_id_date(text: str) -> str:
    # 25 Juli 2025 | 13 Oktober 2025 | 1 Januari 2023
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(20\d{2})", text, flags=re.IGNORECASE)
    if m:
        d, mon, y = m.group(1), m.group(2).lower(), m.group(3)
        mon = ID_MONTHS.get(mon, "01")
        return f"{y}-{mon}-{int(d):02d}"
    # 25/07/2025 atau 25-07-2025
    m = re.search(r"(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](20\d{2})", text)
    if m:
        d, mon, y = m.groups()
        return f"{y}-{int(mon):02d}-{int(d):02d}"
    return ""

def extract_money(s: str) -> str:
    m = re.findall(r"Rp\.?\s*[\d\.\,]+", s, flags=re.IGNORECASE)
    return m[0] if m else ""

# =======================
# Katalog jenis dokumen (dipakai find_jenis_surat)
# =======================
# TYPE_PATTERNS = [
#     ("surat_perintah_tugas", r"\bsurat\s+perintah\s+tugas\b|\bspt\b"),
#     ("surat_perintah_perjalanan_dinas", r"\bsurat\s+perintah\s+perjalanan\s+dinas\b|\bsppd\b"),
#     ("surat_bukti_pengeluaran|belanja", r"\bsurat\s+bukti\s+pengeluaran\b|\bbelanja\b|BKP|BKU"),
#     ("laporan_pertanggungjawaban_bendahara_pengeluaran", r"\blaporan\s+pertanggungjawaban\s+bendahara\s+pengeluaran\b|\bspj\b"),
#     ("register_sp2d", r"\bregister\s+sp2d\b"),
#     ("berita_acara", r"\bberita\s+acara\b|\bbast\b|\bserah\s+terima\b"),
#     ("surat_keputusan", r"\bsurat\s+keputusan\b|\bkeputusan\s+(bupati|kepala)\b|\bsk\b"),
#     ("surat_undangan", r"\bsurat\s+undangan\b|\bundangan\b"),
#     ("nota_dinas", r"\bnota\s+dinas\b|\bnd\b"),
#     ("surat_permohonan", r"\bsurat\s+permohonan\b|\bpermohonan\b"),
#     ("surat_edaran", r"\bsurat\s+edaran\b|\bse\b"),
#     ("surat_pengantar", r"\bsurat\s+pengantar\b"),
#     ("surat_tugas", r"\bsurat\s+tugas\b(?!\s*perintah)"),
#     ("surat_keterangan", r"\bsurat\s+keterangan\b"),
#     ("pengumuman", r"\bpengumuman\b"),
#     ("kontrak_perjanjian", r"\b(perjanjian|kontrak)\b"),
#     ("faktur_kwitansi", r"\bfaktur\b|\bkwitansi\b"),
#     ("dokumen_kementerian", r"\bkementerian\b|\bpanrb\b|\brepublik\s+indonesia\b"),
#     ("dokumen_tabel_keuangan", r"\bkode\s+rekening\b|\bjumlah\s+anggaran\b|\bsisa\s+pagu\b"),
#     ("dokumen_umum", r".*"),  # fallback
# ]
# PRIORITAS HEADING (judul halaman / 10-15 baris pertama)
# ====== Tambahan untuk HEADINGS (judul 10–15 baris pertama) ======
TYPE_PATTERNS_HEAD = [
    # --- Baru ditambahkan ---
    ("surat_pesanan", r"(?im)^\s*(surat\s+pesanan|surat\s+pesanan\s*\(sp\)|sp\s*[:\-]?)\b"),
    ("berita_acara_serah_terima_hasil", r"(?im)^\s*berita\s+acara\s+serah\s+terima\s+hasil\b"),
    ("bukti_penerimaan_negara", r"(?im)^\s*bukti\s+penerimaan\s+negara\b"),
    ("cetakan_kode_billing", r"(?im)^\s*cetakan\s+kode\s+billing\b"),
    ("keputusan_bupati", r"(?im)^\s*(keputusan\s+bupati\s+[a-z\s]*)\b"),

    # --- Yang sudah ada (biarkan) ---
    ("register_sp2d", r"(?im)^\s*(register\s*sp2d)\b"),
    ("spj_belanja", r"(?im)^\s*(laporan\s+pertanggungjawaban|spj\s+belanja|lpj)\b"),
    ("surat_bukti_pengeluaran_belanja", r"(?im)^\s*(surat\s+bukti\s+pengeluaran|surat\s+bukti\s+belanja)\b"),
    ("rincian_biaya_sppd", r"(?im)^\s*(rincian\s+perhitungan\s+biaya\s+perjalanan\s+dinas)\b"),
    ("tanda_terima_sppd", r"(?im)^\s*(tanda\s+terima\s+biaya\s+sppd)\b"),
    ("berita_acara", r"(?im)^\s*(berita\s+acara)\b"),
    ("hasil_evaluasi_spbe", r"(?im)^\s*(hasil\s+evaluasi\s+spbe)\b"),
    ("surat_perintah_perjalanan_dinas", r"(?im)^\s*(surat\s+perintah\s+perjalanan\s+dinas|sppd)\b"),

    # umum
    ("nota_dinas", r"(?im)^\s*(nota[\s\-]*dinas|notadinas|nd\s*[:\-]?)\b"),
    ("surat_permohonan", r"(?im)^\s*(surat\s+permohonan|permohonan)\b"),
    ("surat_undangan", r"(?im)^\s*(surat\s+undangan|undangan)\b"),
    ("surat_pengantar", r"(?im)^\s*(surat\s+pengantar)\b"),
    ("surat_keputusan", r"(?im)^\s*(surat\s+keputusan|keputusan\s+(bupati|kepala)|sk)\b"),
    ("dokumen_kementerian", r"(?im)^\s*(kementerian|panrb|republik\s+indonesia)\b"),
]

# ====== Tambahan untuk BODY (isi dokumen) ======
TYPE_PATTERNS_BODY = [
    # --- Baru ditambahkan ---
    ("surat_pesanan", r"\bsurat\s+pesanan\b|(?<![a-z])sp(?![a-z])\b"),
    ("berita_acara_serah_terima_hasil", r"\bberita\s+acara\s+serah\s+terima\s+hasil\b"),
    ("bukti_penerimaan_negara", r"\bbukti\s+penerimaan\s+negara\b|\bntpn\b|\bntb?n\b|\bstan\b|\bntp\b"),
    ("cetakan_kode_billing", r"\bcetakan\s+kode\s+billing\b|\bkode\s+billing\b|\bid\s+billing\b"),
    ("keputusan_bupati", r"\bkeputusan\s+bupati\s+[a-z\s]*\b"),

    # --- Yang sudah ada (biarkan) ---
    ("register_sp2d", r"\bregister\s*sp2d\b"),
    ("sp2d", r"\bsp2d\b"),
    ("spj_belanja", r"\b(laporan\s+pertanggungjawaban|spj|lpj)\b"),
    ("surat_bukti_pengeluaran_belanja", r"\bsurat\s+bukti\s+(pengeluaran|belanja)\b"),
    ("bast", r"\bberita\s+acara\s+(serah\s+terima|hasil\s+pekerjaan)\b|\bbast\b"),
    ("rincian_biaya_sppd", r"\brincian\s+perhitungan\s+biaya\s+perjalanan\s+dinas\b"),
    ("tanda_terima_sppd", r"\btanda\s+terima\s+biaya\s+sppd\b"),
    ("surat_perintah_perjalanan_dinas", r"\bsurat\s+perintah\s+perjalanan\s+dinas\b|\bsppd\b"),
    ("hasil_evaluasi_spbe", r"\bhasil\s+evaluasi\s+spbe\b|\bnilai\s+indeks\b"),

    # umum
    ("nota_dinas", r"\bnota\s*dinas\b|\bnota-dinas\b|(?<![a-z])nd(?![a-z])"),
    ("surat_permohonan", r"\bsurat\s+permohonan\b|\bpermohonan\b"),
    ("surat_undangan", r"\bsurat\s+undangan\b|\bundangan\b"),
    ("surat_pengantar", r"\bsurat\s+pengantar\b"),
    ("surat_keputusan", r"\bsurat\s+keputusan\b|\bkeputusan\s+(bupati|kepala)\b|\b(?<![a-z])sk(?![a-z])\b"),
    ("dokumen_kementerian", r"\bkementerian\b|\bpanrb\b|\brepublik\s+indonesia\b"),
    ("dokumen_umum", r".*"),
]

def normkey(s: str) -> str:
    # huruf kecil, buang semua non-alfanum supaya tahan spasi/tanda baca/garis miring
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def take_head(text: str, n_lines: int = 15) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return "\n".join(lines[:n_lines])

def find_jenis_surat(text: str) -> str:
    """
    Klasifikasi jenis dokumen berbasis skor:
    - heading (15 baris awal) diberi bobot lebih tinggi
    - ada 'hard lock' untuk judul yang sering menempel (tanpa spasi)
    - ada penalti untuk mencegah salah klasifikasi (ND vs Undangan, SPT vs SPPD, dst.)
    """
    t = text or ""
    head = take_head(t, 15)
    hn = normkey(head)
    tn = normkey(t)

    # ========= HARD LOCKS (langsung pulang kalau match) =========
    hardlocks = [
        ("surat_bukti_pengeluaran_belanja", ["suratbuktipengeluaranbelanja"]),
        ("register_sp2d",                    ["registersp2d"]),
        ("rincian_biaya_sppd",               ["rincianperhitunganbiayaperjalanandinas"]),
        ("tanda_terima_sppd",                ["tandaterimabiayasppd"]),
        ("nota_dinas",                       ["notadinas"]),   # judul menempel
        ("surat_perintah_perjalanan_dinas",  ["suratperintahperjalanandinas","sppd"]),
        ("berita_acara",                     ["beritaacara"]),
        ("keputusan_bupati",                 ["keputusanbupati"]),
        ("bukti_penerimaan_negara",          ["buktipenerimaannegara"]),
        ("cetakan_kode_billing",             ["cetakankodebilling","kodebilling"]),
        ("surat_perintah_tugas", ["suratperintahtugas","suratperintah","spt"]),
        ("spj_belanja", ["laporanpertanggungjawabanbendaharapengeluaran","spjbelanja"]),
    ]
    for label, keys in hardlocks:
        if any(k in hn or k in tn for k in keys):
            return label

    # ========= FITUR & BOBOT =========
    # Skor heading lebih tinggi (H=3) daripada body (B=1)
    H, B = 3.0, 1.0
    features = {
        # Keuangan / Perjalanan dinas
        "register_sp2d": [
            (r"(?im)^\s*register\s*sp2d\b", H), (r"\bregister\s*sp2d\b", B),
        ],
        "spj_belanja": [
            (r"(?im)^\s*(laporan\s+pertanggungjawaban|lpj|spj)\b", H),
            (r"\b(laporan\s+pertanggungjawaban|lpj|spj)\b", B),
        ],
        "surat_bukti_pengeluaran_belanja": [
            (r"(?im)^\s*surat\s*bukti\s*peng[eu]luaran\s*(?:[/\\-]|\s*)\s*belanja\b", H),
            (r"\bsurat\s*bukti\s*peng[eu]luaran\s*(?:[/\\-]|\s*)\s*belanja\b|\bBKP\b|\bBKU\b", B),
        ],
        "rincian_biaya_sppd": [
            (r"(?im)^\s*rincian\s+perhitungan\s+biaya\s+perjalanan\s+dinas\b", H),
            (r"\brincian\s+perhitungan\s+biaya\s+perjalanan\s+dinas\b", B),
        ],
        "tanda_terima_sppd": [
            (r"(?im)^\s*tanda\s+terima\s+biaya\s+sppd\b", H),
            (r"\btanda\s+terima\s+biaya\s+sppd\b", B),
        ],
        "surat_perintah_perjalanan_dinas": [
            (r"(?im)^\s*surat\s+perintah\s+perjalanan\s+dinas\b|\b^\s*sppd\b", H),
            (r"\bsurat\s+perintah\s+perjalanan\s+dinas\b|\bsppd\b", B),
        ],
        "berita_acara": [
            (r"(?im)^\s*berita\s+acara\b", H),
            (r"\bberita\s+acara\s+(serah\s+terima|hasil\s+pekerjaan)?\b|\bbast\b", B),
        ],
        "hasil_evaluasi_spbe": [
            (r"(?im)^\s*hasil\s+evaluasi\s+spbe\b", H), (r"\bhasil\s+evaluasi\s+spbe\b|\bnilai\s+indeks\b", B),
        ],
        "surat_pesanan": [
            (r"(?im)^\s*(surat\s+pesanan|surat\s+pesanan\s*\(sp\)|sp\s*[:\-]?)\b", H),
            (r"\bsurat\s+pesanan\b|(?<![a-z])sp(?![a-z])\b", B),
        ],
        "bukti_penerimaan_negara": [
            (r"(?im)^\s*bukti\s+penerimaan\s+negara\b", H),
            (r"\bbukti\s+penerimaan\s+negara\b|\bntpn\b|\bntb?n\b|\bstan\b|\bntp\b", B),
        ],
        "cetakan_kode_billing": [
            (r"(?im)^\s*cetakan\s+kode\s+billing\b", H),
            (r"\bcetakan\s+kode\s+billing\b|\bkode\s+billing\b|\bid\s+billing\b", B),
        ],
        "surat_perintah_tugas": [
        (r"(?im)^\s*(surat\s+perintah\s+tugas|s\.?p\.?t\.?)\b", H),
        (r"\bsurat\s+perintah\s+tugas\b|\b(?<![a-z])spt(?![a-z])\b", B),
        ],
        "spj_belanja": [
            (r"(?im)^\s*(laporan\s+pertanggungjawaban\s+bendahara\s+pengeluaran|spj\s+belanja)\b", H),
            (r"\b(laporan\s+pertanggungjawaban\s+bendahara\s+pengeluaran|spj\s+belanja|lpj\s+belanja)\b", B),
        ],

        # Surat umum
        "nota_dinas": [
            (r"(?im)^\s*(nota[\s\-]*dinas|notadinas|nd\s*[:\-]?)\b", H),
            (r"\bnota\s*dinas\b|\bnota-dinas\b|(?<![a-z])nd(?![a-z])", B),
        ],
        "surat_undangan": [
            # heading UNDANGAN kuat ⇒ skor besar
            (r"(?im)^\s*(surat\s+undangan|undangan)\b", H),
            # di body wajib ada konteks 'mengundang/harap hadir' agar tak nabrak ND
            (r"\bundangan\b.*\b(harap\s+hadir|mengundang|diundang|kehadiran)\b", B),
        ],
        "surat_permohonan": [
            (r"(?im)^\s*(surat\s+permohonan|permohonan)\b", H),
            (r"\bsurat\s+permohonan\b|\bpermohonan\b", B),
        ],
        "surat_pengantar": [
            (r"(?im)^\s*surat\s+pengantar\b", H), (r"\bsurat\s+pengantar\b", B),
        ],
        "surat_tugas": [
            (r"(?im)^\s*surat\s+tugas\b(?!\s*perintah)", H),
            (r"\bsurat\s+tugas\b(?!\s*perintah)", B),
        ],
        "surat_perintah_tugas": [
            (r"(?im)^\s*surat\s+perintah\s+tugas\b|\b^\s*spt\b", H),
            (r"\bsurat\s+perintah\s+tugas\b|\bspt\b", B),
        ],
        "surat_keputusan": [
            (r"(?im)^\s*surat\s+keputusan\b", H),
            (r"\bsurat\s+keputusan\b|\b(?<![a-z])sk(?![a-z])\b", B),
        ],
        "keputusan_bupati": [
            (r"(?im)^\s*keputusan\s+bupati\b", H), (r"\bkeputusan\s+bupati\b", B),
        ],
        "dokumen_kementerian": [
            (r"(?im)^\s*(kementerian|republik\s+indonesia|panrb)\b", H),
            (r"\bkementerian\b|\bpanrb\b|\brepublik\s+indonesia\b", B),
        ],
    }

    # ========= Hitung skor =========
    scores = {k: 0.0 for k in features.keys()}

    for label, pats in features.items():
        for patt, w in pats:
            if re.search(patt, head, flags=re.IGNORECASE|re.MULTILINE):
                scores[label] += w
            elif re.search(patt, t, flags=re.IGNORECASE):
                scores[label] += w/2  # sedikit bonus jika hanya muncul di body

    # ========= Aturan anti-bentrok (penalti) =========
    def penalize(lbl, amount=2.0): scores[lbl] = max(0.0, scores.get(lbl, 0.0) - amount)

    # ND vs Undangan: kalau heading cocok ND, turunkan Undangan
    if scores["nota_dinas"] >= H: penalize("surat_undangan", 3.0)
    # SPT vs SPPD
    if scores["surat_perintah_tugas"] > 0 and scores["surat_perintah_perjalanan_dinas"] > 0:
        if scores["surat_perintah_perjalanan_dinas"] >= H: penalize("surat_tugas", 2.5)
    # Jika ada label keuangan kuat, turunkan label surat umum
    keu = ["register_sp2d","spj_belanja","surat_bukti_pengeluaran_belanja",
            "rincian_biaya_sppd","tanda_terima_sppd","bukti_penerimaan_negara","cetakan_kode_billing"]
    if any(scores[k] >= H for k in keu):
        for g in ["nota_dinas","surat_undangan","surat_permohonan","surat_pengantar","surat_tugas"]:
            penalize(g, 1.5)

    # ========= Ambil label terbaik =========
    best_label, best_score = max(scores.items(), key=lambda x: x[1])
    # Ambang: harus minimal 2.5 (≈ satu match heading), jika tidak → dokumen_umum
    return best_label if best_score >= 2.5 else "dokumen_umum"


# def find_jenis_surat(text: str) -> str:
#     t = (text or "").lower()
#     for label, patt in TYPE_PATTERNS:
#         if re.search(patt, t, flags=re.IGNORECASE):
#             return label
#     return "dokumen_umum"

# def find_jenis_surat(text: str) -> str:
#     """
#     Deteksi jenis dokumen dari kata kunci di teks (regex-only).
#     Kembalikan token snake_case sederhana; nanti dinormalisasi ke folder.
#     """
#     if not text:
#         return "dokumen_umum"
#     t = text.lower()

#     # urutkan dari yang paling spesifik
#     mapping = [
#         ("lembar_disposisi",   r"\blembar\s+disposisi\b"),
#         ("nota_dinas",         r"\bnota\s+dinas\b|\bnd\b"),  # ND sering muncul di nomor
#         ("surat_undangan",     r"\bsurat\s+undangan\b|\bundangan\b"),
#         ("surat_perintah_tugas", r"\bsurat\s+perintah\s+tugas\b|\bspt\b"),
#         ("surat_tugas",        r"\bsurat\s+tugas\b"),
#         ("surat_edaran",       r"\bsurat\s+edaran\b|\bse\b"),
#         ("berita_acara",       r"\bberita\s+acara\b"),
#         ("laporan",            r"\blaporan\b"),
#         ("surat_pengantar",    r"\bsurat\s+pengantar\b|\bpengantar\b"),
#         ("surat_permohonan",   r"\bsurat\s+permohonan\b|\bpermohonan\b"),
#         ("surat_pemberitahuan", r"\bsurat\s+pemberitahuan\b|\bpemberitahuan\b"),
#         ("surat_keputusan",    r"\bsurat\s+keputusan\b|\bsk\b"),
#         ("surat_peringatan",   r"\bsurat\s+peringatan\b"),
#         ("surat_pengumuman",   r"\bsurat\s+pengumuman\b"),
#         ("surat_resmi",        r"\bsurat\s+resmi\b"),
#         ("surat_bukti_pengeluaran", r"\bsurat\s+bukti\s+pengeluaran\b"),
#         ("laporan_pertanggungjawaban_bendahara_pengeluaran", r"\blaporan\s+pertanggungjawaban\s+bendahara\s+pengeluaran\b"),
#         ("register_sp2d",    r"\bregister\s+sp2d\b"),
#         ("surat_perintah_tugas", r"\bsurat\s+perintah\s+tugas\b|\bspt\b"),
#         ("rincian_perhitungan_biaya_perjalanan_dinas", r"\brincian\s+perhitungan\s+biaya\s+perjalanan\s+dinas\b"),
#         ("tanda_terima_biaya_sppd", r"\btanda\s+terima\s+biaya\s+sppd\b"),
#         ("hasil_evaluasi_spbe", r"\bhasil\s+evaluasi\s+spbe\b"),
#         ("surat_perintah_perjalanan_dinas", r"\bsurat\s+perintah\s+perjalanan\s+dinas\b"),
#         ("surat_pesanan",     r"\bsurat\s+pesanan\b"),
#         ("bukti_penerimaan_negara", r"\bbukti\s+penerimaan\s+negara\b"),
#         ("laporan_pelaksanaan_kegiatan", r"\blaporan\s+pelaksanaan\s+kegiatan\b"),
#         ("keputusan_bupati", r"\bkeputusan\s+bupati\b"),
#         ("SPJ_Belanja_Fungsional", r"\bspj\s+belanja\s+fungsional\b"),
#     ]
#     for label, patt in mapping:
#         if re.search(patt, t, flags=re.IGNORECASE):
#             return label
#     return "dokumen_umum"
# =======================
# Extractor field (per-jenis)
# =======================
COMMON_FIELDS = {
    "nomor_surat": [
        r"(?:Nomor|No\.)\s*[:\-]?\s*([^\n]+)",   # baris setelah label
    ],
    "sifat":       [r"\bsifat\s*[:\-]?\s*([^\n]+)"],
    "lampiran":    [r"\blampiran\s*[:\-]?\s*([^\n]+)"],
    "perihal":     [r"\b(perihal|hal|tentang)\b\s*[:\-]?\s*([^\n]+)"],
    "tanggal_surat":[r"(\d{1,2}\s+[A-Za-z]+\s+20\d{2})", r"(\d{1,2}[\/\-]\d{1,2}[\/\-]20\d{2})"],
    "pembuat":     [r"\b(pejabat|kepala|sekretaris|pimpinan)\b.*?\b([A-Z][A-Z \.']+),?\s*(?:NIP|$)"],
    "nip":         [r"\bNIP\s*[:\-]?\s*([0-9 ]{12,20})"],
    "unit":        [r"\b(dinas|badan|sekretariat|uptd|kementerian)[^\n]{0,60}"],
    "tembusan":    [r"\btembusan\s*[:\-]?\s*(.+)"],
}

TYPE_SPECIFIC = {
    "surat_perintah_tugas": {
        "judul": [r"\bsurat\s+perintah\s+tugas\b[^\n]*"],
        "daftar_nama": [r"\bNama\s*:?.*?(?=UNTUK|Menugaskan|Keterangan|Ditetapkan)",],
    },
    "surat_perintah_perjalanan_dinas": {
        "tujuan": [r"\bke\s*:?\s*([^\n]+)"],
        "lama_hari": [r"\bselama\s*(\d+)\s*hari\b"],
    },
    "register_sp2d": {
        "periode": [r"\bPeriode\s*:\s*([^\n]+)"],
        "unit":    [r"\b(Dinas|Badan|Sekretariat)[^\n]{0,50}"],
        "total":   [r"\bTOTAL\b[^\n]*?([\d\.\,]+)"],
    },
    "laporan_pertanggungjawaban_bendahara_pengeluaran": {
        "bulan": [r"\bBulan\s*[:\-]?\s*([A-Za-z]+)"],
        "program":[r"\bProgram\b\s*:\s*([^\n]+)"],
        "jumlah_anggaran":[r"\bJumlah\s+Anggaran\b[^\n]*?([\d\.\,]+)"],
        "sisa_pagu":[r"\bSisa\s+Pagu\b[^\n]*?([\d\.\,]+)"],
    },
    "surat_bukti_pengeluaran|belanja": {
        "jumlah_rupiah":[r"\bRp\.?\s*[\d\.\,]+", r"sejumlah\s*Rp\.?\s*[\d\.\,]+"],
        "kode_rekening":[r"\bKode\s+Rekening\b\s*[:\-]?\s*([^\n]+)"],
        "untuk_pembayaran":[r"\bUntuk\s*:?\s*([^\n]+)"],
        "penerima":[r"\bYang\s+berhak\s+menerima\s+pembayaran\s*:?\s*([^\n]+)"],
    },
    "berita_acara": {
        "agenda":[r"\bberita\s+acara\s*([^\n]+)"],
        "tanggal_acara":[r"tanggal\s*[:\-]?\s*([^\n]+)"],
    },
    "surat_keputusan": {
        "tentang":[r"\btentang\b\s*[:\-]?\s*([^\n]+)"],
    },
    "surat_undangan": {
        "acara":[r"\bundangan\b[^\n]*", r"\bacara\s*[:\-]?\s*([^\n]+)"],
        "waktu":[r"\bwaktu\s*[:\-]?\s*([^\n]+)"],
        "tempat":[r"\btempat\s*[:\-]?\s*([^\n]+)"],
    },
    "dokumen_tabel_keuangan": {
        "kode_rekening":[r"\bKode\s+Rekening\b"],
        "jumlah_anggaran":[r"\bJumlah\s+Anggaran\b[^\n]*?([\d\.\,]+)"],
    },
    # tambahkan sesuai kebutuhan…
}

def extract_fields_by_patterns(text: str, doc_type: str) -> dict:
    """Scan teks pakai COMMON_FIELDS + TYPE_SPECIFIC[doc_type]. Ambil nilai pertama yang ketemu."""
    fields = {}
    # common
    for k, patts in COMMON_FIELDS.items():
        for p in patts:
            m = re.search(p, text, flags=re.IGNORECASE)
            if m:
                g = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)
                # khusus 'perihal' kadang pattern punya 2 group (label+isi)
                if k == "perihal" and m.lastindex and m.lastindex >= 2:
                    g = m.group(2)
                fields[k] = g.strip()
                break
    # type-specific
    spec = TYPE_SPECIFIC.get(doc_type, {})
    for k, patts in spec.items():
        for p in patts:
            m = re.search(p, text, flags=re.IGNORECASE|re.DOTALL)
            if m:
                g = m.group(1) if m.lastindex else m.group(0)
                fields[k] = " ".join(g.strip().split())
                break

    # normalisasi nilai yang umum
    if "tanggal_surat" in fields:
        iso = parse_id_date(fields["tanggal_surat"])
        if iso: fields["tanggal_surat_iso"] = iso
    if "jumlah_rupiah" in fields:
        fields["jumlah_rupiah_norm"] = extract_money(fields["jumlah_rupiah"])
    if "total" in fields:
        fields["total"] = fields["total"].replace(".", "").replace(",", ".")
    return fields

def normalize_text(s: str) -> str:
    if not s:
        return ""
    # buang spasi/tab berlebih
    s = re.sub(r"[ \t]+", " ", s)
    # hapus baris kosong
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    return "\n".join(lines)
def is_letterhead_line(ln: str) -> bool:
    """True jika baris tampak seperti kop/head (instansi, alamat, kontak, dsb)."""
    if not ln:
        return False
    if HEADER_RE.search(ln):
        return True
    # baris ALL CAPS panjang → sangat mungkin kop / judul instansi
    cap_ratio = sum(1 for c in ln if c.isupper()) / max(1, sum(1 for c in ln if c.isalpha()))
    if cap_ratio > 0.85 and len(ln) >= 12:
        return True
    # terlalu pendek atau nomor telepon/website
    if re.search(r"\b\d{3,}[-.\s]?\d{3,}", ln):  # pola telp umum
        return True
    if re.search(r"https?://|www\.", ln):
        return True
    return False

def strip_letterhead(text: str) -> str:
    """Buang baris yang terlihat seperti kop/identitas kantor di awal dokumen."""
    lines = normalize_text(text).splitlines()
    cleaned = []
    seen_substantive = False
    for ln in lines:
        if not seen_substantive:
            # sampai ketemu baris substantif (bukan kop)
            if is_letterhead_line(ln):
                continue
            # baris dekoratif (=====, ------)
            if re.fullmatch(r"[-=_.]{4,}", ln):
                continue
            # lewat sini berarti ln cukup substantif
            seen_substantive = True
        cleaned.append(ln)
    return "\n".join(cleaned)
# ---------------------------
# Perihal extractor (preserve symbols)
# ---------------------------
def _fix_inline_spacing(s: str) -> str:
    # unescape HTML (&amp; → &)
    s = html.unescape(s or "")
    # tambah spasi di sekitar & dan /
    s = re.sub(r"\s*&\s*", " & ", s)
    s = re.sub(r"\s*/\s*", " / ", s)
    # pecah CamelCase nempel: "PermohonanNama" → "Permohonan Nama"
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    # kompres spasi
    s = re.sub(r"[ \t]+", " ", s).strip()
    # buang leading ":" ganda dari OCR
    s = re.sub(r"^:+\s*", "", s)
    return s

def find_perihal(text: str, image_path: Path = None) -> str:
    """Ambil Perihal/Hal/Tentang, mempertahankan &, -, /, dll. Hanya merapikan spasi."""
    if not text:
        return "tanpa perihal"
    t = normalize_for_fields(text)

    # baris label
    m = re.search(r"(?im)^(?:Perihal|Hal|Tentang)\s*:\s*(.+)$", t)
    if m:
        first = m.group(1).strip()
        # join 1 baris berikutnya kalau terlalu pendek atau tampak “tergantung”
        lines = t.splitlines()
        idx = next((i for i, ln in enumerate(lines) if re.search(r"(?im)^(?:Perihal|Hal|Tentang)\s*:\s*", ln)), None)
        nxt = (lines[idx+1].strip() if idx is not None and idx+1 < len(lines) else "")
        cand = (first + (" " + nxt if len(first) < 12 or first.endswith(("&", "-", "/")) else "")).strip()
    else:
        # fallback: ambil kalimat awal yang paling “judul”
        cand = next((ln.strip() for ln in t.splitlines() if len(ln.strip()) > 10), "")

    # rapikan spasi internal, tapi jangan hilangkan simbol seperti & dan -
    raw = _fix_inline_spacing(cand)

    return raw or "tanpa perihal"

# ---------------------------
# Slug khusus untuk nama file (pertahankan '-')
# ---------------------------
def perihal_to_slug(s: str, max_len: int = 120) -> str:
    s = html.unescape(s or "").strip()
    s = _fix_inline_spacing(s)
    # untuk nama file: izinkan huruf/angka/._- dan spasi → underscore
    s = re.sub(r'[^A-Za-z0-9.\- _/&()]', '_', s)
    # ganti & dan / supaya path aman (kalau mau pertahankan &, hapus baris ini)
    s = s.replace("&", " dan ").replace("/", " ")
    s = re.sub(r"[ \t]+", " ", s).strip().replace(" ", "_")
    s = re.sub(r"_{2,}", "_", s)
    return (s[:max_len] or "tanpa_perihal")

# def find_perihal(text: str, image_path: Path = None) -> str:
#     """
#     Ambil perihal dari teks penuh TANPA bergantung posisi/crop.
#     1) Prioritas baris yang mengandung 'Perihal/Hal/Tentang'
#     2) Jika tidak ada, pilih baris substantif (bukan kop/salam) yang terlihat seperti subjek.
#     """
#     if not text:
#         return "tanpa_perihal"

#     t = strip_letterhead(text)
#     lines = [ln for ln in t.splitlines() if ln.strip()]

#     # 1) Cari kata kunci langsung
#     key_patterns = [
#         r"\bperihal\b\s*[:\-]?\s*(.+)",
#         r"\bhal\b\s*[:\-]?\s*(.+)",
#         r"\btentang\b\s*[:\-]?\s*(.+)",
#     ]
#     for ln in lines:
#         for p in key_patterns:
#             m = re.search(p, ln, flags=re.IGNORECASE)
#             if m:
#                 raw = m.group(1).strip()
#                 raw = re.sub(r"[^A-Za-z0-9\s]", " ", raw)
#                 words = raw.split()
#                 return ("_".join(words[:20]).lower() or "tanpa_perihal")

#     # 2) Heuristik baris subjek (hindari salam & tujuan surat)
#     STOPLINE = re.compile(
#     r"^(ass?alamu|yth\.?|kepada|kepada\s*yth|dengan\s*hormat|menindaklanjuti|sehubungan|berdasarkan)\b",
#     re.IGNORECASE
#     )

#     KEY_HINTS = re.compile(
#         r"rapat|undangan|permohonan|pemberitahuan|pengantar|laporan|koordinasi|klarifikasi|konfirmasi|nota|dinas|edaran|perintah|tugas|disposisi|keputusan|penugasan|pembinaan|pembayaran|tagihan|perbaikan|perawatan|pengaduan",
#         re.IGNORECASE
#     )

#     candidates = []
#     for ln in lines[:40]:  # batasi ke 40 baris pertama
#         if is_letterhead_line(ln):
#             continue
#         if STOPLINE.search(ln):
#             continue
#         # buang baris terlalu pendek atau terlalu teknis (no telp/dll)
#         if len(ln) < 12:
#             continue
#         if not re.search(r"[A-Za-z]{3,}", ln):
#             continue
#         # punya kata kunci subjek?
#         score = 0
#         if KEY_HINTS.search(ln):
#             score += 2
#         # baris satu kalimat relatif “judul”
#         if ln.endswith("."):
#             pass
#         else:
#             score += 1
#         # bukan ALL CAPS total
#         cap_ratio = sum(1 for c in ln if c.isupper()) / max(1, sum(1 for c in ln if c.isalpha()))
#         if cap_ratio < 0.85:
#             score += 1
#         # simpan kandidat
#         if score >= 2:
#             candidates.append((score, ln))

#     if candidates:
#         candidates.sort(reverse=True)
#         raw = candidates[0][1]
#         raw = re.sub(r"[^A-Za-z0-9\s]", " ", raw)
#         words = raw.split()
#         return ("_".join(words[:20]).lower() or "tanpa_perihal")

#     # 3) fallback: 6 kata pertama yang “cukup kata”
#     words = re.findall(r"[A-Za-z0-9]{3,}", t)
#     return "_".join(words[:6]).lower() if words else "tanpa_perihal"


def short_title_from_text(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text)
    return "_".join(words[:20]) if words else "dokumen"

def normalize_field_linebreaks(s: str) -> str:
    # Satukan label yang patah baris (Nomor/No/Perihal/Hal/Tentang)
    s = re.sub(r"(?:Nomor|No\.?|Perihal|Hal|Tentang)\s*[:\-–]?\s*\n+\s*",
               lambda m: m.group(0).split()[0].rstrip(":") + ": ",
               s, flags=re.IGNORECASE)
    s = re.sub(r"[ \t]+", " ", s)
    return s

def save_text_sidecar(src_path: Path, text: str) -> Path:
    txt_path = src_path.with_suffix(".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text or "")
    return txt_path

def save_meta_sidecar(src_img: Path, fields: dict) -> Path:
    import json
    meta_path = src_img.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(fields, f, ensure_ascii=False, indent=2)
    return meta_path

def safe_rename(path: Path, new_stem: str) -> Path:
    new_stem = sanitize_for_filename(new_stem, max_len=180)
    dst = path.with_name(f"{new_stem}{path.suffix.lower()}")
    counter = 1
    while dst.exists() and dst.resolve() != path.resolve():
        dst = path.with_name(f"{new_stem}_{counter}{path.suffix.lower()}")
        counter += 1
    path.rename(dst)
    return dst

def sanitize_for_filename(s: str, max_len: int = 200) -> str:
    import unicodedata
    if not s:
        return "noname"
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("/", "-").replace("\\", "-")
    s = re.sub(r'[:*?"<>|]+', "_", s)
    s = re.sub(r"[^A-Za-z0-9\.\-_ ]+", "_", s)
    s = re.sub(r"[\s_]+", "_", s)
    s = s.strip("_.- ")
    if len(s) > max_len:
        s = s[:max_len]
    if not s:
        s = "noname"
    return s


def get_download_link(data, filename, text, file_type="auto"):
    if hasattr(data, "save"):
        buffered = io.BytesIO()
        data.save(buffered, format="PNG")
        data_bytes = buffered.getvalue()
        mime = "image/png"
    elif isinstance(data, (str, bytes, io.IOBase)):
        if isinstance(data, str):
            with open(data, "rb") as f:
                data_bytes = f.read()
            mime = mimetypes.guess_type(data)[0] or "application/octet-stream"
        elif isinstance(data, bytes):
            data_bytes = data
            mime = "application/octet-stream"
        else:
            data_bytes = data.read()
            mime = "application/octet-stream"
    else:
        raise TypeError("data harus berupa PIL.Image, path file, bytes, atau IO stream")

    b64 = base64.b64encode(data_bytes).decode()
    href = f'<a href="data:{mime};base64,{b64}" download="{filename}">{text}</a>'
    return href

# ==========================
# Google Drive helpers (FIND-ONLY root & year) + mirror folder lokal
# ==========================
# ==========================
# Google Drive (OAuth user, bukan service account)
# ==========================
BASE_DIR = Path(__file__).resolve().parent
CLIENT_SECRET_JSON = str(BASE_DIR / "client_secret.json")
SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME = "application/vnd.google-apps.folder"


def build_drive_service():
    """
    Autentikasi ke Google Drive sebagai USER (OAuth).
    Token disimpan di token.json supaya tidak perlu login berulang.
    """
    creds = None
    token_path = BASE_DIR / "token.json"

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRET_JSON, SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as token:
            token.write(creds.to_json())

    return build("drive", "v3", credentials=creds)


def _escape_name(name: str) -> str:
    """Escape tanda petik untuk query Drive."""
    return name.replace("'", r"\'")


def drive_find_only_root(drive, name: str) -> str:
    """
    Cari folder bernama `name` di seluruh Drive (tidak dibatasi root).
    Ambil satu yang pertama.
    """
    q = (
        f"name = '{_escape_name(name)}' "
        f"and mimeType = '{FOLDER_MIME}' and trashed = false"
    )
    res = drive.files().list(
        q=q,
        spaces="drive",
        fields="files(id, name, parents)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
    ).execute()
    files = res.get("files", [])

    if not files:
        raise RuntimeError(
            f"Folder root '{name}' tidak ditemukan di Drive. "
            f"Pastikan sudah dibuat & Anda punya akses."
        )

    if len(files) > 1:
        st.warning(
            f"Ditemukan {len(files)} folder bernama '{name}'. "
            f"Menggunakan folder pertama dengan id: {files[0]['id']}"
        )

    return files[0]["id"]


def ensure_child_folder_by_name(drive, parent_id: str, name: str) -> str:
    """Pastikan ada subfolder `name` di bawah parent_id, kalau belum ada maka dibuat."""
    q = (
        f"'{parent_id}' in parents and "
        f"name = '{_escape_name(name)}' and "
        f"mimeType = '{FOLDER_MIME}' and trashed = false"
    )
    res = drive.files().list(
        q=q,
        spaces="drive",
        fields="files(id, name)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
    ).execute()
    files = res.get("files", [])
    if files:
        return files[0]["id"]

    file_metadata = {
        "name": name,
        "mimeType": FOLDER_MIME,
        "parents": [parent_id],
    }
    created = drive.files().create(
        body=file_metadata,
        fields="id",
        supportsAllDrives=True,
    ).execute()
    return created["id"]


def upload_file_to_drive(drive, local_path: Path, parent_id: str):
    """Upload satu file ke folder Drive tertentu."""
    file_metadata = {
        "name": local_path.name,
        "parents": [parent_id],
    }
    media = MediaFileUpload(str(local_path), resumable=True)
    try:
        drive.files().create(
            body=file_metadata,
            media_body=media,
            fields="id",
            supportsAllDrives=True,
        ).execute()
    except HttpError as e:
        st.error(f"Drive API error: {e}")


def mirror_local_tree_to_drive_year(local_root: Path, drive_root_name: str, selected_year: str):
    """
    Mirror struktur folder `local_root` di bawah:
    drive_root_name / selected_year / (subfolder2 dst...)
    + progress bar Streamlit.
    """
    drive = build_drive_service()

    # 0. Hitung total file yang akan diupload (untuk progress bar)
    all_files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(local_root):
        for fname in filenames:
            all_files.append(Path(dirpath) / fname)

    total_files = len(all_files)
    if total_files == 0:
        st.warning("Tidak ada file yang akan diupload ke Google Drive.")
        return

    progress = st.progress(0)
    status = st.empty()
    uploaded = 0

    # 1. Cari folder root (misal 'Arsip Digital')
    root_id = drive_find_only_root(drive, drive_root_name)

    # 2. Pastikan folder tahun di dalamnya
    year_id = ensure_child_folder_by_name(drive, root_id, str(selected_year))

    # 3. Mirror seluruh pohon folder + update progress
    for dirpath, dirnames, filenames in os.walk(local_root):
        rel = Path(dirpath).relative_to(local_root)
        if str(rel) == ".":
            current_parent_id = year_id
            rel_display = Path(".")
        else:
            parts = list(rel.parts)
            current_parent_id = year_id
            for part in parts:
                current_parent_id = ensure_child_folder_by_name(
                    drive, current_parent_id, part
                )
            rel_display = rel

        for fname in filenames:
            fpath = Path(dirpath) / fname
            uploaded += 1
            # update status + progress
            status.text(
                f"☁️ Upload {uploaded}/{total_files} "
                f"→ {drive_root_name}/{selected_year}/{rel_display / fname}"
            )
            upload_file_to_drive(drive, fpath, current_parent_id)
            progress.progress(int(uploaded / total_files * 100))

    # 4. Selesai
    progress.progress(100)
    status.text(f"✅ Upload ke Google Drive selesai ({total_files} file).")




# ==========================
# Streamlit UI
# ==========================
STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / "static"
DOWNLOADS_PATH = STREAMLIT_STATIC_PATH / "downloads"
# Streamlit Cloud biasanya punya env var ini
IS_CLOUD = os.environ.get("STREAMLIT_RUNTIME") == "cloud" or os.path.exists("/mount/src")

BASE_DIR = Path("/tmp/digiarsip") if IS_CLOUD else Path(".")
DOWNLOADS_PATH = BASE_DIR / "downloads"

DOWNLOADS_PATH.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 384
preprocess_transforms = image_preprocess_transforms()

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.title("Batch Document Scanner")

uploaded_files = st.file_uploader(
    "Upload document images (png/jpg/jpeg). You can select many files",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

method = st.radio("Select model:", ("MobilenetV3-Large", "Resnet-50"), horizontal=True)

col_left, col_right = st.columns((1, 1))

with col_left:
    st.markdown("**Settings**")
    max_preview = st.number_input("Show preview of first N results", min_value=1, max_value=12, value=4)
    process_button = st.button("Start processing")
    st.markdown("**Google Drive**")
    drive_root_name = st.text_input(
        "Nama folder root di Drive (harus sudah ada & dishare ke service account):",
        value="Arsip Digital"
    )
    upload_choice = st.selectbox(
        "Upload ke Drive?",
        ["Tidak upload", "2023", "2024", "2025"],
        index=0
    )

with col_right:
    st.markdown("**Model**")
    st.caption("Model will be loaded once and reused for all images.")

if uploaded_files:
    st.info(f"{len(uploaded_files)} files selected. Click **Start processing** to run.")
    names_preview = [f.name for f in uploaded_files[:20]]
    st.write("Files (first 20):", names_preview)

if process_button and uploaded_files:
    model_name = "mbv3" if method == "MobilenetV3-Large" else "r50"
    model = load_model(model_name=model_name)  # cached resource

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        processed_paths = []
        pdf_paths = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        total = len(uploaded_files)
        for i, uploaded_file in enumerate(uploaded_files, start=1):
            status_text.text(f"Processing {i}/{total}: {uploaded_file.name}")

            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                st.warning(f"Failed to read {uploaded_file.name}, skipping.")
                progress_bar.progress(int(i/total*100))
                continue

            # === Warp ===
            warped = scan(image_true=img, trained_model=model, image_size=IMAGE_SIZE)

            # === Enhance (Auto, no sliders) ===
            enhanced = enhance_doc(
                warped,
                grayscale=False,
                clip_limit=3.0,
                tile=8,
                sharpen_amount=1.0,
            )

            # Save PNG
            safe_name = Path(uploaded_file.name).stem
            out_img_path = tmpdir / f"{safe_name}_scanned.png" if not grayscale else tmpdir / f"{safe_name}_scanned.png"
            if grayscale and len(enhanced.shape) == 2:
                cv2.imwrite(str(out_img_path), enhanced)
            else:
                cv2.imwrite(str(out_img_path), enhanced, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            processed_paths.append(out_img_path)

            # Save single-page PDF
            pil_img = Image.fromarray(cv2.cvtColor(enhanced if len(enhanced.shape)==3 else cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB))
            out_pdf_path = tmpdir / f"{safe_name}_scanned.pdf"
            pil_img.convert("RGB").save(out_pdf_path, "PDF", resolution=300.0)
            pdf_paths.append(out_pdf_path)

            progress_bar.progress(int(i/total*100))

        status_text.text("Creating ZIP archive...")

        # Create consolidated ZIPs
        zip_all_path = tmpdir / "scanned_all_results.zip"
        with zipfile.ZipFile(zip_all_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in processed_paths + pdf_paths:
                zf.write(p, arcname=p.name)

        zip_pdf_path = tmpdir / "scanned_pdfs_only.zip"
        with zipfile.ZipFile(zip_pdf_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in pdf_paths:
                zf.write(p, arcname=p.name)

        zip_img_path = tmpdir / "scanned_images_only.zip"
        with zipfile.ZipFile(zip_img_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in processed_paths:
                zf.write(p, arcname=p.name)

        status_text.text("✅ Done — auto-crop + enhance + exports ready.")

        # Preview (Before → After)
        st.subheader("Preview (Before → After)")
        cols = st.columns(min(max_preview, len(processed_paths)))
        for i, p in enumerate(processed_paths[:max_preview]):
            with cols[i]:
                a, b = st.columns(2)
                with a:
                    st.caption("Before")
                    st.image(uploaded_files[i], use_container_width=True)
                with b:
                    st.caption("After")
                    st.image(str(p), use_container_width=True)
                st.caption(Path(p).name)

        # # =====================
        # # OCR + grouping folders
        # # =====================
        # st.info("🔍 Deteksi teks, nomor, jenis surat, dan perihal...")

        # ocr_output_dir = tmpdir / "grouped_docs"
        # ocr_output_dir.mkdir(exist_ok=True)
        # group_map = {}
        # auto_counter = {}

        # def get_autonumber(jenis: str):
        #     if jenis not in auto_counter:
        #         auto_counter[jenis] = 1
        #     else:
        #         auto_counter[jenis] += 1
        #     return f"{auto_counter[jenis]:03d}"

        # processed_paths = [p for p in processed_paths if Path(p).exists()]
        # pdf_paths = [p for p in pdf_paths if Path(p).exists()]
        # def normalize_doc_type(jenis: str) -> str:
        #     """
        #     Ubah jenis jadi label folder & prefix file:
        #     - lowercase, hilangkan underscore/spasi
        #     """
        #     if not jenis:
        #         return "dokumen umum"
        #     j = jenis.lower().replace("_", "_").replace(" ", "_")
        #     return j if j else "dokumen umum"
        # for img_path, pdf_path in zip(processed_paths, pdf_paths):
        #     # 1) Ekstrak teks penuh (Docling)
        #     txt_img = extract_text_from_image(Path(img_path))     # sudah grayscale->Docling
        #     # txt_pdf = extract_text_from_pdf(Path(pdf_path))       # Docling juga
        #     full_text = normalize_text((txt_img + "\n").strip())
        #     full_text = normalize_field_linebreaks(full_text)
        #     print(txt_img)
            
        #     # 2) Deteksi berbasis TEKS (tanpa posisi/crop)
        #     nomor   = find_nomor_surat(full_text)
        #     jenis   = find_jenis_surat(full_text)
        #     perihal = find_perihal(full_text)

        #     # 3) Sanitasi & build nama
        #     jenis_safe   = sanitize_for_filename(jenis)
        #     docname      = normalize_doc_type(jenis_safe)   # -> folder + prefix file
        #     nomor_safe   = sanitize_for_filename(nomor)
        #     perihal_safe = sanitize_for_filename(perihal)

        #     # autonumber per jenis jika nomor kosong
        #     def get_autonumber_per_doc(docname_key: str):
        #         if docname_key not in auto_counter:
        #             auto_counter[docname_key] = 1
        #         else:
        #             auto_counter[docname_key] += 1
        #         return f"{auto_counter[docname_key]:03d}"

        #     nomor_part = nomor_safe if nomor_safe else get_autonumber_per_doc(docname)

        #     # Nama file: namadokumen_nomor_perihal
        #     # --- Buat nama dasar ---
        #     if nomor_safe:
        #         base_name = "_".join([p for p in [jenis_safe, nomor_safe, perihal_safe] if p])
        #     else:
        #         base_name = "_".join([p for p in [jenis_safe, perihal_safe] if p])

        #     # --- Subfolder: ocr_output_dir / jenis / base_name ---
        #     sub_folder = ocr_output_dir / jenis_safe / base_name
        #     sub_folder.mkdir(parents=True, exist_ok=True)

        #     def safe_copy(src_path, dst_folder, base_name):
        #         if not src_path:
        #             return None
        #         src = Path(src_path)
        #         if not src.exists():
        #             print(f"⚠️ File hilang, dilewati: {src}")
        #             return None
        #         dst_folder.mkdir(parents=True, exist_ok=True)
        #         ext = src.suffix.lower() or ".png"
        #         base_name = sanitize_for_filename(base_name, max_len=200)
        #         dst = dst_folder / f"{base_name}{ext}"
        #         counter = 1
        #         while dst.exists():
        #             dst = dst_folder / f"{base_name}_{counter}{ext}"
        #             counter += 1
        #         try:
        #             shutil.copy2(str(src), str(dst))
        #             return dst
        #         except Exception as e:
        #             st.error(f"❌ Gagal menyalin {src} -> {dst}: {e}")
        #             return None

        #     saved_files = []
        #     for src in [img_path, pdf_path]:
        #         saved = safe_copy(src, sub_folder, base_name)
        #         if saved:
        #             saved_files.append(saved)

        #     group_map.setdefault(str(sub_folder.relative_to(ocr_output_dir)), []).extend(saved_files)
        # =====================
        # Tahap A — Ekstrak teks (Docling) → simpan .txt (tanpa rename)
        # =====================
        st.subheader("Tahap A — Ekstrak teks dokumen")
        progress_a = st.progress(0)
        status_a = st.empty()
        text_index = []  # list of dict: {"img": , "pdf": , "txt": }
        total_a = len(processed_paths)
        if total_a == 0:
            st.warning("Tidak ada berkas untuk Tahap A (processed_paths kosong).")
        else:
            for i, (img_path, pdf_path) in enumerate(zip(processed_paths, pdf_paths)):
                # update awal (sebelum kerja berat) biar terlihat bergerak
                pct = int(i / total_a * 100)
                progress_a.progress(pct)
                status_a.text(f"🔍 Ekstraksi teks {i+1}/{total_a}: {Path(img_path).name}")

                try:
                    raw_text = extract_text_from_image(Path(img_path))
                    full_text = normalize_field_linebreaks(normalize_text((raw_text or "").strip()))
                    txt_path = save_text_sidecar(Path(img_path), full_text)
                    # simpan index untuk tahap berikutnya
                    text_index.append({"img": img_path, "pdf": pdf_path, "txt": txt_path})
                except Exception as e:
                    st.error(f"Gagal memproses {Path(img_path).name}: {e}")
                finally:
                    # pastikan bar naik meski error
                    pct_done = int((i + 1) / total_a * 100)
                    progress_a.progress(min(100, pct_done))

            # kunci di 100% pada akhir tahap
            progress_a.progress(100)
            status_a.text(f"✅ Tahap A selesai — {len(text_index)} dokumen diekstrak.")

        
        # =====================
        # Tahap B — Metadata dari teks (multi-jenis) + progress
        # =====================
        st.subheader("Tahap B — Analisis metadata")
        progress_b = st.progress(0)
        status_b = st.empty()

        records = []
        total_b = len(text_index)
        if total_b == 0:
            st.warning("Tidak ada berkas untuk Tahap B (text_index kosong).")
        else:
            for i, rec in enumerate(text_index):
                # update awal
                pct = int(i / total_b * 100)
                progress_b.progress(pct)
                status_b.text(f"🧩 Analisis metadata {i+1}/{total_b}: {Path(rec['img']).name}")

                try:
                    with open(rec["txt"], "r", encoding="utf-8") as f:
                        text_all = f.read()

                    # deteksi jenis / nomor / perihal (fungsi milikmu)
                    doc_type = find_jenis_surat(text_all) or "dokumen_umum"
                    nomor    = find_nomor_surat(text_all) or ""
                    # perihal  = find_perihal(text_all) or "tanpa_perihal"
                    perihal_raw  = find_perihal(text_all) or "tanpa perihal"
                    perihal_raw  = _fix_inline_spacing(perihal_raw)   # rapikan spasi & unescape
                    perihal_safe = perihal_to_slug(perihal_raw)

                    # field tambahan per-jenis (jika kamu pakai extractor tambahan)
                    extra = extract_fields_by_patterns(text_all, doc_type) if 'extract_fields_by_patterns' in globals() else {}

                    jenis_safe   = sanitize_for_filename(doc_type)
                    nomor_safe   = sanitize_for_filename(nomor)
                    # perihal_safe = sanitize_for_filename(perihal)

                    meta = {
                        "source_image": Path(rec["img"]).name,
                        "source_pdf":   Path(rec["pdf"]).name,
                        "text_file":    Path(rec["txt"]).name,
                        "jenis_dokumen": doc_type,
                        "nomor_surat":   nomor,
                        "perihal":       perihal_raw,
                        **extra,
                        "jenis_safe":   jenis_safe,
                        "nomor_safe":   nomor_safe,
                        "perihal_safe": perihal_safe,
                        "rename_policy":"jenis_nomor_perihal if nomor else jenis_perihal",
                    }
                    meta_path = save_meta_sidecar(Path(rec["img"]), meta)
                    records.append({**rec, "meta": meta_path, **meta})

                except Exception as e:
                    st.error(f"Metadata gagal untuk {Path(rec['img']).name}: {e}")

                finally:
                    progress_b.progress(int((i+1) / total_b * 100))

            progress_b.progress(100)
            status_b.text(f"✅ Tahap B selesai — {len(records)} metadata dibuat.")
            st.success("Tahap B OK.")


        for r in records[:max_preview]:
            st.write(f"🖼️ {r['source_image']} | 📄 {r['source_pdf']}")
            st.write(f"• jenis: **{r['jenis_dokumen']}**")
            st.write(f"• nomor: **{r['nomor_safe'] or '—'}**")
            st.write(f"• perihal: **{r['perihal_safe']}**")
            st.caption(f"TXT: {r['text_file']}  |  META: {Path(r['meta']).name}")
            st.markdown("---")
        # =====================
        # Tahap C — Susun struktur folder + ZIP + tombol download (langsung)
        # =====================
        st.subheader("Tahap C — Susun struktur & buat ZIP")
        progress_c = st.progress(0)
        status_c = st.empty()

        output_root = Path(tmpdir) / "grouped_by_type"
        output_root.mkdir(exist_ok=True)

        def build_stem(jenis_safe: str, nomor_safe: str, perihal_safe: str) -> str:
            parts = [jenis_safe]
            if nomor_safe:
                parts.append(nomor_safe)
            parts.append(perihal_safe)
            return sanitize_for_filename("_".join([p for p in parts if p]), max_len=180)

        total_c = len(records)
        if total_c == 0:
            st.warning("Tidak ada berkas untuk Tahap C (records kosong).")
        else:
            for i, r in enumerate(records):
                # update awal
                progress_c.progress(int(i / total_c * 100))
                status_c.text(f"📂 Menata {i+1}/{total_c}: {Path(r['img']).name}")

                try:
                    stem = build_stem(r["jenis_safe"], r["nomor_safe"], r["perihal_safe"])
                    jenis_dir = output_root / r["jenis_safe"]
                    sub_dir   = jenis_dir / stem
                    sub_dir.mkdir(parents=True, exist_ok=True)

                    sources = [
                        Path(r["img"]),
                        Path(r["pdf"]),
                        Path(r["txt"]),
                        Path(r["img"]).with_suffix(".meta.json"),
                    ]
                    for src in sources:
                        src = Path(src)
                        if not src.exists():
                            continue
                        dst = sub_dir / f"{stem}{src.suffix.lower()}"
                        counter = 1
                        while dst.exists():
                            dst = sub_dir / f"{stem}_{counter}{src.suffix.lower()}"
                            counter += 1
                        shutil.copy2(str(src), str(dst))

                except Exception as e:
                    st.error(f"Gagal menata {Path(r['img']).name}: {e}")

                finally:
                    progress_c.progress(int((i+1) / total_c * 100))

            # buat ZIP
            status_c.text("🧵 Mengarsipkan menjadi ZIP…")
            final_zip_path = Path(tmpdir) / "grouped_by_type.zip"
            try:
                with zipfile.ZipFile(final_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for f in output_root.rglob("*"):
                        if f.is_file():
                            zf.write(f, arcname=str(f.relative_to(output_root)))
            except Exception as e:
                st.error(f"Gagal membuat ZIP: {e}")
            finally:
                progress_c.progress(100)
                status_c.text("✅ Tahap C selesai — ZIP siap diunduh.")

            # tombol download
            with open(final_zip_path, "rb") as f:
                st.download_button(
                    "📥 Download Struktur Folder (ZIP)",
                    data=f.read(),
                    file_name="grouped_by_type.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
            st.success("ZIP siap diunduh.")
        # === Upload ke Google Drive dalam run yang sama ===
        if upload_choice != "Tidak upload":
            try:
                st.info(f"Mengupload struktur folder ke Drive: {drive_root_name}/{upload_choice}")
                mirror_local_tree_to_drive_year(
                    local_root=output_root,          # akar folder lokal yg barusan dibuat
                    drive_root_name=drive_root_name,
                    selected_year=upload_choice
                )
            except Exception as e:
                st.error(f"Gagal upload ke Drive: {e}")

        # tombol download langsung
        # with open(final_zip_path, "rb") as f:
        #     st.download_button(
        #         "📦 Download Struktur Folder (ZIP)",
        #         data=f.read(),
        #         file_name="grouped_by_type.zip",
        #         mime="application/zip",
        #         use_container_width=True,
        #     )






        # final_zip_path = tmpdir / "scanned_grouped_hierarchy.zip"
        # with zipfile.ZipFile(final_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        #     for file in ocr_output_dir.rglob("*"):
        #         if file.is_file():
        #             zf.write(file, arcname=str(file.relative_to(ocr_output_dir)))

        # st.success(f"✅ Struktur folder lengkap dengan nomor & perihal selesai dibuat ({len(group_map)} subfolder).")

        # for folder, files in group_map.items():
        #     st.markdown(f"📂 **{folder}** — {len(files)} file")
        #     for f in files[:1]:
        #         if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        #             st.image(str(f), width=240)

        # col1, col2, col3, col4 = st.columns(4)
        # with col1:
        #     with open(zip_all_path, "rb") as f:
        #         st.download_button(
        #             label=f"📦 Download All Results (PDF + Image) [{len(pdf_paths)+len(processed_paths)} files]",
        #             data=f.read(),
        #             file_name="scanned_all_results.zip",
        #             mime="application/zip",
        #             use_container_width=True,
        #         )
        # with col2:
        #     with open(zip_pdf_path, "rb") as f:
        #         st.download_button(
        #             label=f"🖨️ Download All PDFs Only ({len(pdf_paths)} files)",
        #             data=f.read(),
        #             file_name="scanned_pdfs_only.zip",
        #             mime="application/zip",
        #             use_container_width=True,
        #         )
        # with col3:
        #     with open(zip_img_path, "rb") as f:
        #         st.download_button(
        #             label=f"🖼️ Download All Images Only ({len(processed_paths)} files)",
        #             data=f.read(),
        #             file_name="scanned_images_only.zip",
        #             mime="application/zip",
        #             use_container_width=True,
        #         )
        # with col4:
        #     with open(final_zip_path, "rb") as f:
        #         st.download_button(
        #             "📦 Download Folder Berdasarkan Jenis, Nomor & Perihal (ZIP)",
        #             data=f.read(),
        #             file_name="scanned_grouped_hierarchy.zip",
        #             mime="application/zip",
        #             use_container_width=True,
        #         )

        # status_text.text("✅ Processing complete!")
else:
    if not uploaded_files:
        st.info("No files uploaded yet.")