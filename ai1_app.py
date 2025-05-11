import streamlit as st
from PIL import Image
import os
import requests
import io
from io import BytesIO
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import time
import cv2
from tensorflow.keras.models import load_model
import google.generativeai as genai
import pytesseract
import tensorflow as tf
import torch
from transformers import pipeline
import gdown




# تعريف الخدمات الطبية مع روابط صور
SERVICES = {
    "Chest X-ray": {
        "description": "AI-powered chest X-ray analysis",
        "page": "chest_xray_page",
        "image": "https://images.stockcake.com/public/1/1/3/113c0212-bbec-4994-b370-7965a4157df9_large/doctor-reviews-x-ray-stockcake.jpg"
    },
    "Brain Tumor Diagnosis": {
        "description": "Brain MRI tumor detection",
        "page": "brain_tumor_page",
        "image": "https://www.intercoastalmedical.com/wp-content/uploads/sites/494/2020/04/iStock-1199813214.jpg"
    },
    "Liver Disease Diagnosis": {
        "description": "Liver fibrosis imaging and analysis",
        "page": "liver_page",
        "image": "https://static.biospace.com/dims4/default/1c559b7/2147483647/strip/true/crop/622x350+2+0/resize/1000x563!/format/webp/quality/90/?url=https%3A%2F%2Fk1-prod-biospace.s3.us-east-2.amazonaws.com%2Fbrightspot%2Flegacy%2FBioSpace-Assets%2FE601C051-24B9-4C00-B352-954397EAEF32.jpg"
    },
    "Analysis": {
        "description": "Automated test interpretation",
        "page": "analysis_page",
        "image": "https://img.freepik.com/premium-photo/healthcare-team-bustling-hospital-with-doctor-standing-out-holding-folder_207634-13088.jpg"
    },
    "Eye Scan": {
        "description": "Retinal image analysis",
        "page": "eye_scan_page",
        "image": "https://www.virginia-lasik.com/wp-content/uploads/2021/01/Retinal-Exam-Nguyen--2048x1360.jpg"
    },
    "Fracture Detection": {
        "description": "Bone fracture identification",
        "page": "fracture_page",
        "image": "https://wp02-media.cdn.ihealthspot.com/wp-content/uploads/sites/309/2022/05/iStock-840336238-1024x576.jpg"
    },
}

# عدد الخدمات المعروضة في كل صفحة
SERVICES_PER_PAGE = 3


def load_image_from_url(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except:
        return None


def main():
    # تهيئة حالة الصفحة
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0

    # حساب عدد الصفحات الإجمالية
    total_pages = (len(SERVICES) + SERVICES_PER_PAGE - 1) // SERVICES_PER_PAGE

    # عرض الصفحة الرئيسية أو صفحة الخدمة
    if 'selected_service' not in st.session_state:
        show_main_page(total_pages)
    else:
        show_service_page(st.session_state.selected_service)


def show_main_page(total_pages):
    # تصميم الصفحة الرئيسية
    st.markdown("""
    <style>

      .stApp {
        background-color: #f0f2f6;
    }

    .title-container {
        text-align: center;
        margin-bottom: 30px;
    }
    .title-text {
        font-size: 2.5rem;
        color: #2c3e50;
        font-weight: bold;
    }
    .service-card {
        border: 1px solid #e0e0e0;
        border-radius: 15px;
        padding: 0;
        margin: 15px;
        text-align: center;
        transition: transform 0.3s, box-shadow 0.3s;
        background-color: #f9f9f9;
        height: 380px;
        display: flex;
        flex-direction: column;
        position: relative;
        overflow: hidden;
    }
    .service-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        border: 1px solid #4CAF50;
    }
    .service-image-container {
        width: 100%;
        height: 220px;
        overflow: hidden;
    }
    .service-image {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.5s;
    }
    .service-card:hover .service-image {
        transform: scale(1.05);
    }
    .service-content {
        padding: 15px;
        flex-grow: 1;
        display: flex;
        flex-direction: column;
    }
    .service-title {
        font-weight: bold;
        margin: 5px 0;
        color: #2c3e50;
        font-size: 1.2rem;
    }
    .service-description {
        font-size: 0.95rem;
        color: #7f8c8d;
        margin-bottom: 15px;
        flex-grow: 1;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 8px 15px;
        cursor: pointer;
        transition: background-color 0.3s;
        width: 100%;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .centered-container {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        margin: 30px 0;
    }
    .navigation-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 20px;
        padding: 0 20px;
    }
    .page-indicator {
        color: #7f8c8d;
        font-size: 0.9rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # العنوان في منتصف الصفحة
    st.markdown("""
    <div class="title-container">
        <div class="title-text">Our Medical AI Services</div>
        <div style="color: #7f8c8d;">Select a service to get started</div>
    </div>
    """, unsafe_allow_html=True)

    # حساب الخدمات التي يجب عرضها في الصفحة الحالية
    start_idx = st.session_state.current_page * SERVICES_PER_PAGE
    end_idx = min(start_idx + SERVICES_PER_PAGE, len(SERVICES))
    current_services = list(SERVICES.items())[start_idx:end_idx]

    # عرض بطاقات الخدمات في منتصف الصفحة
    cols = st.columns(len(current_services))
    for idx, (service_name, service_info) in enumerate(current_services):
        with cols[idx]:
            st.markdown(f"""
            <div class="service-card">
                <div class="service-image-container">
                    <img class="service-image" src="{service_info["image"]}" alt="{service_name}">
                </div>
                <div class="service-content">
                    <div class="service-title">{service_name}</div>
                    <div class="service-description">{service_info["description"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Streamlit button that will navigate to the service page
            if st.button(f"Select", key=f"select_{service_name}"):
                st.session_state.selected_service = service_name
                st.rerun()

    # أزرار التنقل بين الصفحات باستخدام Streamlit buttons
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        prev_disabled = st.session_state.current_page == 0
        if st.button("← Previous", disabled=prev_disabled, key="prev_button"):
            st.session_state.current_page -= 1
            st.rerun()

    with col2:
        st.markdown(f"""
        <div class="page-indicator">
            Page {st.session_state.current_page + 1} of {total_pages}
        </div>
        """, unsafe_allow_html=True)

    with col3:
        next_disabled = st.session_state.current_page == total_pages - 1
        if st.button("Next →", disabled=next_disabled, key="next_button"):
            st.session_state.current_page += 1
            st.rerun()


def show_service_page(service_name):
    st.title(f"🔬 {service_name}")
    st.markdown(f"{SERVICES[service_name]['description']}")

    # عرض صورة الخدمة
    st.image(SERVICES[service_name]["image"], use_container_width=True)


    if service_name == "Chest X-ray":
        chest_xray_page()
    elif service_name == "Brain Tumor Diagnosis":
        brain_tumor_page()
    elif service_name == "Liver Disease Diagnosis":
        liver_page()
    elif service_name == "Analysis":
        analysis_page()
    elif service_name == "Eye Scan":
        eye_scan_page()
    elif service_name == "Fracture Detection":
        fracture_page()

    if st.button("← Back to Services"):
        del st.session_state.selected_service
        st.rerun()


def chest_xray_page():
    @st.cache_resource
    def load_model():
        files = {
            "config.json": {
                "file_id": "14M2rmv00uGCT7xbq7nHu7jkUaSsTQ5OG",
                "output": "config.json"
            },
            "model.safetensors": {
                "file_id": "1v90JJcPsad13gtxMluqCRau5HBmonjUH",
                "output": "model.safetensors"
            },
            "preprocessor_config.json": {
                "file_id": "1ycZG5YhATFS67-zODHZhLNY8WE7hphH9",
                "output": "preprocessor_config.json"
            }
        }

        model_dir = "chest_xray_model"
        os.makedirs(model_dir, exist_ok=True)

        try:
            for file_name, file_info in files.items():
                output_path = os.path.join(model_dir, file_name)

                if not os.path.exists(output_path):
                    st.info(f"Downloading {file_name}...")
                    gdown.download(
                        f"https://drive.google.com/uc?id={file_info['file_id']}",
                        output_path,
                        quiet=False
                    )

                if not os.path.exists(output_path):
                    st.error(f"❌ Failed to download: {file_name}")
                    return None

            st.success("✅ Model files loaded successfully.")

            return pipeline(
                "image-classification",
                model=model_dir,
                device="cpu"
            )

        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

    # تحميل النموذج
    model = load_model()

    if model is None:
        st.warning("Model couldn't be loaded.")
    else:
        uploaded_file = st.file_uploader("Upload an X-ray image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file).convert("RGB")
                st.image(img, caption="Uploaded Image", use_container_width=True)

                if st.button("🔍 Analyze Image"):
                    with st.spinner("Analyzing..."):
                        predictions = model(img)
                        top_prediction = predictions[0]
                        label = top_prediction['label']
                        score = top_prediction['score'] * 100

                        st.markdown(f"### 🩺 Diagnosis: *{label}*")
                        st.markdown(f"*Confidence:* {score:.2f}%")

            except Exception as e:
                st.error(f"❌ Error analyzing image: {str(e)}")

                
def brain_tumor_page():
    @st.cache_resource
    def load_models():
        try:
            # تحميل فقط نموذج التصنيف من Google Drive
            file_id = "1nTRy7Kn5nHDlAuXoB3ffFhwiV1I3KTRg"
            output = "brain_classification_model.h5"

            if not os.path.exists(output):
                gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

            if not os.path.exists(output):
                st.error("فشل تحميل نموذج التصنيف")
                return None

            classification_model = tf.keras.models.load_model(output)

            return classification_model

        except Exception as e:
            st.error(f"حدث خطأ أثناء تحميل النموذج: {str(e)}")
            return None

    classification_model = load_models()

    st.write("### Upload Brain MRI")
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        try:
            st.write("تحميل نموذج الكشف YOLOv5 من الملفات المحلية...")
            brain_detection_model = torch.hub.load(
                './yolov5-master',  # المسار المحلي إلى مجلد yolov5
                'custom',
                path='brain_detection_model.pt',  # ملف النموذج
                source='local',
                force_reload=True
            )
        except Exception as e:
            st.error(f"فشل تحميل نموذج YOLOv5: {e}")
            st.stop()

        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)

        # تطبيق نموذج الكشف
        results = brain_detection_model(image_np)
        result_img = np.squeeze(results.render())

        st.image(image, caption="Uploaded MRI", use_container_width=True)

        # تجهيز الصورة للتصنيف
        img = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        if st.button("Detect Tumors"):
            with st.spinner('Processing...'):
                brain_classes = ['No_Tumor', 'Tumor']
                prediction = classification_model.predict(img_array)
                predicted_class = int(prediction[0][0] > 0.5) if prediction.shape[1] == 1 else np.argmax(prediction)

                st.markdown(f"### Diagnosis: **{brain_classes[predicted_class]}**")
                st.image(result_img, caption="Detection Result", use_container_width=True)

                
                
########################################################
# def brain_tumor_page():
#     @st.cache_resource  # تخزين النموذج في الذاكرة للجلسات المتعددة
#     def load_models():
#         try:
#             # رابط Google Drive المعدل (استبدل ?usp=sharing بـ &export=download)
#             file_id = "1nTRy7Kn5nHDlAuXoB3ffFhwiV1I3KTRg"
#             output = "brain_classification_model.h5"

#             dfile_id = '17gbv9iQuW1wFBFNN_dFCI41104LQW1ed'
#             doutput = 'brain_detection_model.pt'

#             # تحميل الملف
#             gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
#             gdown.download(f"https://drive.google.com/uc?id={dfile_id}", doutput, quiet=False)

#             if not os.path.exists(output):
#                 return None, None

#             if not os.path.exists(doutput):
#                 st.error("فشل تحميل نموذج الكشف YOLOv5")
#                 return None, None

#             # تحميل النماذج
#             classification_model = tf.keras.models.load_model(output)

#             return classification_model, doutput

#         except Exception as e:
#             st.error(f"error: {str(e)}")
#             return None, None

#     classification_model, detection_model_path = load_models()

#     st.write("### Upload Brain MRI")
#     uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

#     if uploaded_file is not None:
#             #####################
#         st.write(f"Torch version: {torch.__version__}")
#         st.write("Trying to load YOLOv5 model using torch.hub...")
        

#         try:
#             brain_detection_model = torch.hub.load(
#                 'ultralytics/yolov5',
#                 'custom',
#                 path=detection_model_path,
#                 force_reload=True,
#                 device='cpu'
#                 )
#         except Exception as e:
#             st.error(f"فشل تحميل نموذج YOLOv5: {e}")
#             st.stop()

#         #____________________________
        
#         #brain_detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path=detection_model_path, force_reload=True)
#         image = Image.open(uploaded_file).convert('RGB')
#         image_np = np.array(image)
        
#         results = brain_detection_model(image_np)
#         result_img = np.squeeze(results.render())
        
#         st.image(image, caption="Uploaded MRI", use_container_width=True)
#         img = image.resize((224, 224))
#         img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
#         if st.button("Detect Tumors"):
#             with st.spinner('Processing...'):
#                 brain_classes = ['No_Tumor', 'Tumor']
#                 prediction = classification_model.predict(img_array)
#                 predicted_class = int(prediction[0][0] > 0.5) if prediction.shape[1] == 1 else np.argmax(prediction)
#                 st.markdown(f" Diagnosis: {brain_classes[predicted_class]}")
#                 st.image(result_img, caption="Detection Result", use_container_width=True)
#_________________________________________


def liver_page():
    @st.cache_resource  # تخزين النموذج في الذاكرة للجلسات المتعددة
    def load_models():
        try:
            # رابط Google Drive المعدل (استبدل ?usp=sharing بـ &export=download)
            file_id = "1lOBTOBoDCEtndw5RAOCoRukUtYgMc8xF"
            output = "liver_classification_model.h5"

            dfile_id = '1x1FICVqQMrpBrFqAXJ4woslVrSmPHxML'
            doutput = 'liver_detection_model.pt'

            # تحميل الملف
            gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
            gdown.download(f"https://drive.google.com/uc?id={dfile_id}", doutput, quiet=False)

            if not os.path.exists(output):
                return None, None

            if not os.path.exists(doutput):
                st.error("فشل تحميل نموذج الكشف YOLOv5")
                return None, None

            # تحميل النماذج
            classification_model = tf.keras.models.load_model(output)

            return classification_model, doutput

        except Exception as e:
            st.error(f"error: {str(e)}")
            return None, None

    classification_model, detection_model_path = load_models()
    st.write("### Upload liver Scan")
    view_mode = st.radio("Choose display type:", ["Classification Only", "Detection Only"])
    uploaded_file = st.file_uploader("Upload a radiology image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        liver_detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path= detection_model_path, force_reload=True)
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        results = liver_detection_model(image_np)
        result_img = np.squeeze(results.render())
        st.image(image, caption="Uploaded Image", use_container_width=True)
        img = image.resize((128, 128))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        if st.button("Analyze Scan"):
            with st.spinner('Examining...'):
                liver_classes = ['No_Fibrosis', 'Fibrosis']
                if view_mode == "Classification Only":
                    prediction = classification_model.predict(img_array)
                    predicted_class = int(prediction[0][0] > 0.5) if prediction.shape[1] == 1 else np.argmax(prediction)
                    st.markdown(f" Diagnosis: {liver_classes[predicted_class]}")
                if view_mode == "Detection Only":
                    st.image(result_img, caption="Detection Result", use_container_width=True)


def analysis_page():
    @st.cache_resource
    def load_model():
        genai.configure(api_key="AIzaSyBsA6ixodO7_ODrKGV6kRqiswdN_3n958A")
        return genai.GenerativeModel(model_name="gemini-2.0-flash")

    model = load_model()

    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    st.header("تحميل صورة التحليل")
    uploaded_file = st.file_uploader("قم بتحميل صورة التحليل الطبي", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.image(image, caption="الصورة المحملة", width=400)

        if st.button("تحليل الصورة"):
            with st.spinner("جاري معالجة التحليل..."):

                extracted_text = pytesseract.image_to_string(image, lang='ara+eng')

                # عرض النص المستخرج في مربع نص قابل للطي
                with st.expander("النص المستخرج من الصورة"):
                    st.text_area("النص المستخرج:", value=extracted_text, height=200)

                messages = [
                    {
                        "role": "user",
                        "parts": [f"""أنت مساعد طبي ذكي.
    مهمتك هي:
    - قراءة نتائج التحاليل المرفقة.
    - تلخيص النتائج بشكل تقرير طبي واضح.
    - تقديم نصيحة طبية عامة حسب النتائج.
    - اقتراح التخصص الطبي المناسب إن لزم.

    نتائج التحاليل المستخرجة:
    {extracted_text}"""]
                    }
                ]

                try:
                    response = model.generate_content(messages)

                    # عرض النتائج في مربع بتصميم جميل
                    st.success("تم تحليل الصورة بنجاح!")
                    st.subheader("نتائج التحليل:")
                    st.markdown(f"""
                    <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; border: 1px solid #ccc;">
                        {response.text}
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"حدث خطأ أثناء تحليل الصورة: {str(e)}")
                    st.info(
                        "قد يكون السبب هو تجاوز حدود الاستخدام في واجهة برمجة Google Gemini API. يرجى المحاولة لاحقًا.")

    st.sidebar.header("تعليمات الاستخدام")
    st.sidebar.markdown("""
    1. قم بتحميل صورة التحليل الطبي
    2. انقر على زر "تحليل الصورة"
    3. انتظر حتى تظهر النتائج
    """)

    st.sidebar.header("حول الخدمة")
    st.sidebar.info("""
    هذه الخدمة تستخدم:
    - Google Gemini AI لتحليل النتائج
    - Tesseract OCR لاستخراج النص من الصور
    - Streamlit لواجهة المستخدم

    يمكن قراءة النصوص العربية والإنجليزية من صور التحاليل الطبية وتقديم تحليل لها.
    """)

def eye_scan_page():
    @st.cache_resource  # تخزين النموذج في الذاكرة للجلسات المتعددة
    def load_model():
        # رابط Google Drive المعدل (استبدل ?usp=sharing بـ &export=download)
        url = "https://drive.google.com/uc?id=1sACluiNwV__kosazzRVN-42vnHijYUBK&export=download"
        output = "cnn_model.h5"

        # حذف الملف القديم إذا كان موجوداً
        if os.path.exists(output):
            os.remove(output)

        # تحميل الملف
        gdown.download(url, output, quiet=False)

        # التحقق من وجود الملف قبل التحميل
        if not os.path.exists(output):
            st.error("فشل تحميل ملف النموذج!")
            return None

        try:
            return tf.keras.models.load_model(output)
        except Exception as e:
            st.error(f"خطأ في تحميل النموذج: {str(e)}")
            return None

    model = load_model()
    st.write("قم بتحميل صورة OCT للكشف عن أمراض العين")

    st.sidebar.header("معلومات عن المرض")
    disease_info = {
        "CNV": """
        **التشعب الوعائي المشيمائي**
        - نمو غير طبيعي للأوعية الدموية تحت الشبكية
        - قد يسبب تسرب السوائل وتلف الشبكية
        - مرتبط بالضمور البقعي الرطب المرتبط بالعمر
        """,

        "DME": """
        **وذمة البقعة الصفراء السكرية**
        - تورم في منطقة البقعة الصفراء بسبب مرض السكري
        - يحدث بسبب تسرب السوائل من الأوعية الدموية التالفة
        - قد يؤدي إلى فقدان الرؤية المركزية
        """,

        "DRUSEN": """
        **دروزن**
        - رواسب صفراء صغيرة تحت شبكية العين
        - علامة مبكرة على الضمور البقعي المرتبط بالعمر
        - قد لا تسبب أعراض في المراحل المبكرة
        """,

        "NORMAL": """
        **شبكية طبيعية**
        - شبكية العين صحية بدون علامات مرضية
        - طبقات الشبكية منتظمة وخالية من التورم أو الترسبات
        """
    }

    selected_disease = st.sidebar.selectbox("اختر المرض لمعرفة المزيد", list(disease_info.keys()))
    st.sidebar.markdown(disease_info[selected_disease])


    classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

    def predict_image(img):

        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.reshape(img, (1, 128, 128, 3))

        prediction = model.predict(img)
        class_idx = np.argmax(prediction)
        predicted_class = classes[class_idx]

        return predicted_class, prediction[0]

    uploaded_file = st.file_uploader("قم بتحميل صورة OCT", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        st.image(opencv_image, channels="BGR", caption="الصورة المحملة", use_container_width=True)

        if st.button("تصنيف الصورة"):
            with st.spinner('جاري التصنيف...'):

                if model is not None:
                    predicted_class, all_probs = predict_image(opencv_image)

                    st.success(f"التصنيف: {predicted_class}")

                    st.subheader("معلومات عن الحالة المتوقعة:")
                    st.markdown(disease_info[predicted_class])

                else:
                    st.error("لم يتم تحميل النموذج بشكل صحيح")

    st.markdown("---")
    st.subheader("كيفية استخدام الخدمة")
    st.markdown("""
    1. قم بتحميل صورة OCT باستخدام زر "Browse files"
    2. اضغط على زر "تصنيف الصورة" للحصول على النتيجة
    """)

    st.subheader("ملاحظات مهمة")
    st.markdown("""
    - هذه الخدمة للأغراض التعليمية فقط ولا يغني عن استشارة الطبيب
    - دقة التشخيص تعتمد على جودة الصورة المدخلة
    - يجب التأكد من أن الصورة المحملة هي صورة OCT حقيقية
    """)



def fracture_page():
    @st.cache_resource
    def load_model():
            yolo_model_file_id = "1_gKzNIMSSMBH_uHPz4eq9QMrPIXceDzf"
            model_output = "best.pt"

            try:
                gdown.download(f"https://drive.google.com/uc?id={yolo_model_file_id}",
                               model_output, quiet=False)


                if not os.path.exists(model_output):
                    return None

                from ultralytics import YOLO
                model = YOLO(model_output)

                return model

            except Exception as e:
               return None


    model = load_model()


    if model is None:
        return
    st.write("### Upload Bone X-ray")
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # تحميل الصورة باستخدام PIL
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded X-ray",use_container_width=True)

        if st.button("Check for Fractures"):
            with st.spinner('Analyzing...'):
                # تحويل الصورة إلى صيغة يمكن لـ OpenCV التعامل معاها
                img_array = np.array(img)
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                # حفظ الصورة مؤقتًا لاستخدامها مع YOLO
                temp_img_path = "temp_image.jpg"
                cv2.imwrite(temp_img_path, img_rgb)


                # تشغيل النموذج على الصورة
                predictions = model.predict(source=temp_img_path, save=False, conf=0.1)

                # استخراج الصورة المعالجة
                fracture_detected = False
                for r in predictions:
                    if r.boxes:  # لو فيه كسور تم اكتشافها
                        fracture_detected = True
                    im_show = r.plot(labels=True, boxes=True)  # الصورة مع الكسور مرسومة

                # تحويل الصورة المعالجة إلى صيغة يمكن عرضها في Streamlit
                im_show_rgb = cv2.cvtColor(im_show, cv2.COLOR_BGR2RGB)

                # عرض الصورة المعالجة فقط
                st.image(im_show_rgb, caption="YOLOv8 Prediction", use_container_width=True)

                # عرض نتيجة التحليل
                st.success("Analysis complete!")
                if fracture_detected:
                    st.write("*Results:* Fractures detected!")
                else:
                    st.write("*Results:* No fractures detected")

                # حذف الصورة المؤقتة
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)



if __name__ == "__main__":
    main()
