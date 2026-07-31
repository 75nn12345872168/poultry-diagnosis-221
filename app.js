// ============================================================
// تطبيق تشخيص أمراض الدواجن - يعمل بالكامل داخل المتصفح (Client-Side)
// النموذج: EfficientNet-B0 مُصدَّر إلى صيغة ONNX
// نسخة موثوقية محسّنة: معايرة ثقة (Temperature Scaling) + TTA + سجل ملاحظات محلي
// ============================================================

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.min.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";

// --------------------------------------------------------------
// إعدادات ثابتة (قيم افتراضية احتياطية - تُستبدل تلقائياً بقيم class_config.json إن وُجد)
// --------------------------------------------------------------
const MODEL_PATH = "model/poultry_classifier.onnx";
const CONFIG_PATH = "model/class_config.json";
const IMG_SIZE = 224;
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

const CLASS_NAMES = ["Coccidiosis", "Salmonella", "Newcastle", "Healthy", "Other"];

let CONFIDENCE_THRESHOLD = 0.60;
let TEMPERATURE = 1.0;
let MODEL_VERSION = "غير معروف";
let TEST_ACCURACY = null;

// بوابة الرفض في فضاء التمثيل (Feature-Space Novelty Gate) — خط دفاع مستقل تماماً عن
// نتيجة الـ softmax: أي صورة بعيدة إحصائياً عن كل الفئات الحقيقية (حتى لو لم يرها النموذج
// إطلاقاً بأي شكل) تُرفض تلقائياً. القيم من class_config.json (نفسها المحسوبة في Block 7e).
let NOVELTY_GATE = null; // { real_classes, embedding_dim, class_means, global_var, threshold }

const CLASS_NAMES_AR = {
  Coccidiosis: "كوكسيديا (Coccidiosis)",
  Salmonella: "سالمونيلا (Salmonella)",
  Newcastle: "نيوكاسل (Newcastle Disease)",
  Healthy: "سليم (Healthy)",
  Other: "غير متعلق بالدواجن",
};

const EXPLANATION_TEMPLATES = {
  Coccidiosis: "قوام/لون الفضلات (احتمال وجود دم أو مخاط) مطابق لنمط بصري شبيه بحالات كوكسيديا في بيانات التدريب. هذه مطابقة بصرية، وليست تشخيصاً سريرياً كاملاً.",
  Salmonella: "قوام/لون/تماسك الفضلات مطابق لنمط بصري شبيه بحالات سالمونيلا في بيانات التدريب. هذه مطابقة بصرية، وليست تشخيصاً سريرياً كاملاً.",
  Newcastle: "نمط الفضلات مطابق لنمط بصري شبيه بحالات نيوكاسل في بيانات التدريب. تنبيه مهم: مرض نيوكاسل غالباً ما يصاحبه أيضاً أعراض تنفسية وعصبية لا يستطيع هذا النموذج (المعتمد على صورة الفضلات فقط) رؤيتها أو تقييمها — الفحص البيطري للطائر نفسه ضروري.",
  Healthy: "لم يُلاحَظ نمط فضلات غير طبيعي مقارنة بصور التدريب.",
  Other: "الصورة لا تحتوي على دجاجة أو فضلات أو أعراض مرضية واضحة. يرجى التقاط صورة أوضح تتضمن الدجاجة أو الفضلات مباشرة.",
  Inconclusive: "النموذج غير واثق بدرجة كافية من هذه النتيجة. يرجى إعادة التقاط الصورة بإضاءة أفضل وزاوية أقرب ووضوح أعلى.",
};

let session = null;
let selectedImageEl = null;
let lastResultForFeedback = null;

const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const modelMeta = document.getElementById("model-meta");
const modelVersionFooter = document.getElementById("model-version-footer");
const diagnoseBtn = document.getElementById("diagnose-btn");

async function loadConfig() {
  try {
    const res = await fetch(CONFIG_PATH, { cache: "no-store" });
    if (!res.ok) throw new Error("config not found");
    const cfg = await res.json();
    if (typeof cfg.confidence_threshold === "number") CONFIDENCE_THRESHOLD = cfg.confidence_threshold;
    if (typeof cfg.temperature === "number" && cfg.temperature > 0) TEMPERATURE = cfg.temperature;
    if (cfg.model_version) MODEL_VERSION = cfg.model_version;
    if (typeof cfg.test_accuracy === "number") TEST_ACCURACY = cfg.test_accuracy;
    if (cfg.novelty_gate && Array.isArray(cfg.novelty_gate.global_var)) {
      NOVELTY_GATE = cfg.novelty_gate;
    } else {
      console.warn("لا توجد بوابة رفض (novelty_gate) في class_config.json — سيعتمد التطبيق على فئة Other وحدها.");
    }
    console.info("تم تحميل class_config.json:", cfg);
  } catch (err) {
    console.warn("لم يتم العثور على class_config.json — استخدام القيم الافتراضية (بدون معايرة ثقة).", err);
  }
  modelMeta.textContent = TEST_ACCURACY != null
    ? `إصدار النموذج: ${MODEL_VERSION} · دقة على بيانات الاختبار: ${(TEST_ACCURACY * 100).toFixed(1)}%`
    : `إصدار النموذج: ${MODEL_VERSION}`;
  modelVersionFooter.textContent = `إصدار النموذج: ${MODEL_VERSION}`;
}

async function loadModel() {
  await loadConfig();
  try {
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["wasm"],
    });
    statusDot.classList.add("ready");
    statusText.textContent = "النموذج جاهز — يعمل بدون إنترنت";
  } catch (err) {
    console.error("Model load failed:", err);
    statusDot.classList.add("error");
    statusText.textContent = "تعذّر تحميل النموذج. تأكد من وجود model/poultry_classifier.onnx";
  }
}
loadModel();

const cameraBtn = document.getElementById("camera-btn");
const galleryBtn = document.getElementById("gallery-btn");
const cameraInput = document.getElementById("camera-input");
const galleryInput = document.getElementById("gallery-input");
const previewWrap = document.getElementById("image-preview-wrap");
const previewImg = document.getElementById("image-preview");
const placeholderIcon = document.getElementById("placeholder-icon");

cameraBtn.addEventListener("click", () => cameraInput.click());
galleryBtn.addEventListener("click", () => galleryInput.click());

function handleFileSelect(e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (ev) => {
    previewImg.src = ev.target.result;
    previewImg.style.display = "block";
    placeholderIcon.style.display = "none";
    previewWrap.classList.add("has-image");
    selectedImageEl = previewImg;

    diagnoseBtn.disabled = !session;
    diagnoseBtn.textContent = session ? "🔍 تشخيص الصورة" : "جاري تحميل النموذج...";

    document.getElementById("result-card").classList.remove("visible");
    resetFeedbackUI();
  };
  reader.readAsDataURL(file);
}
cameraInput.addEventListener("change", handleFileSelect);
galleryInput.addEventListener("change", handleFileSelect);

function preprocessImage(imgEl, { flip = false, centerCropFactor = 1.0 } = {}) {
  const canvas = document.createElement("canvas");
  canvas.width = IMG_SIZE;
  canvas.height = IMG_SIZE;
  const ctx = canvas.getContext("2d");
  // نضبط جودة التصغير صراحة (بدل الاعتماد على افتراضي المتصفح) لتقريب سلوك Canvas من
  // خوارزمية cv2/Albumentations (INTER_LINEAR) المستخدمة في التدريب قدر الإمكان.
  // هذا يقلل - لكن لا يُلغي تماماً - فرق التصغير بين المتصفح والتدريب (راجع MODEL_CARD.md).
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";

  const iw = imgEl.naturalWidth || imgEl.width;
  const ih = imgEl.naturalHeight || imgEl.height;

  let sx = 0, sy = 0, sw = iw, sh = ih;
  if (centerCropFactor < 1.0) {
    sw = iw * centerCropFactor;
    sh = ih * centerCropFactor;
    sx = (iw - sw) / 2;
    sy = (ih - sh) / 2;
  }

  if (flip) {
    ctx.translate(IMG_SIZE, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(imgEl, sx, sy, sw, sh, 0, 0, IMG_SIZE, IMG_SIZE);
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  const imageData = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE).data;

  const channelSize = IMG_SIZE * IMG_SIZE;
  const floatData = new Float32Array(3 * channelSize);

  for (let i = 0; i < channelSize; i++) {
    const r = imageData[i * 4] / 255;
    const g = imageData[i * 4 + 1] / 255;
    const b = imageData[i * 4 + 2] / 255;

    floatData[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    floatData[channelSize + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    floatData[2 * channelSize + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }

  return new ort.Tensor("float32", floatData, [1, 3, IMG_SIZE, IMG_SIZE]);
}

function softmaxWithTemperature(logits, temperature) {
  const scaled = logits.map((v) => v / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function confidenceLabel(prob) {
  if (prob >= 0.8) return { label: "عالية", key: "high" };
  if (prob >= 0.55) return { label: "متوسطة", key: "medium" };
  return { label: "منخفضة", key: "low" };
}

const TTA_VARIANTS = [
  { flip: false, centerCropFactor: 1.0 },
  { flip: true, centerCropFactor: 1.0 },
  { flip: false, centerCropFactor: 0.9 },
];

async function runTTAInference(imgEl) {
  const numClasses = CLASS_NAMES.length;
  const accumProbs = new Array(numClasses).fill(0);
  let accumEmbedding = null;

  for (const variant of TTA_VARIANTS) {
    const inputTensor = preprocessImage(imgEl, variant);
    const results = await session.run({ input: inputTensor });
    const logits = Array.from(results.output.data);
    const probs = softmaxWithTemperature(logits, TEMPERATURE);
    for (let i = 0; i < numClasses; i++) accumProbs[i] += probs[i];

    // نجمع الـ embedding أيضاً (المخرج الثاني للنموذج) لتطبيق بوابة الرفض لاحقاً.
    // لو النموذج قديم (مخرج واحد فقط)، results.embedding هتكون undefined وهنتجاهل البوابة بأمان.
    if (results.embedding) {
      const emb = results.embedding.data;
      if (!accumEmbedding) accumEmbedding = new Float64Array(emb.length);
      for (let i = 0; i < emb.length; i++) accumEmbedding[i] += emb[i];
    }
  }

  const probs = accumProbs.map((v) => v / TTA_VARIANTS.length);
  const embedding = accumEmbedding ? Array.from(accumEmbedding, (v) => v / TTA_VARIANTS.length) : null;
  return { probs, embedding };
}

// مسافة (Mahalanobis قُطرية) بين تمثيل الصورة وأقرب فئة حقيقية معروفة. نفس الحساب بالضبط
// المُنفَّذ في Block 7e من النوتبوك، بحيث تكون النتيجة في التطبيق مطابقة للنوتبوك تماماً.
function noveltyScore(embedding) {
  if (!NOVELTY_GATE || !embedding) return null;
  const { class_means, global_var } = NOVELTY_GATE;
  let minDist = Infinity;
  for (const cls of Object.keys(class_means)) {
    const mean = class_means[cls];
    let d = 0;
    for (let i = 0; i < mean.length; i++) {
      const diff = embedding[i] - mean[i];
      d += (diff * diff) / global_var[i];
    }
    if (d < minDist) minDist = d;
  }
  return minDist;
}

diagnoseBtn.addEventListener("click", async () => {
  if (!session || !selectedImageEl) return;

  diagnoseBtn.disabled = true;
  diagnoseBtn.innerHTML = 'جاري التحليل (3 قراءات لدقة أعلى)...<span class="spinner"></span>';

  try {
    const { probs, embedding } = await runTTAInference(selectedImageEl);

    let maxIdx = probs.indexOf(Math.max(...probs));
    let predictedClass = CLASS_NAMES[maxIdx];
    let confidence = probs[maxIdx];

    // بوابة الرفض في فضاء التمثيل: تُطبَّق بعد الـ softmax وتتجاوزه إذا لزم الأمر.
    // أي صورة بعيدة إحصائياً عن كل الفئات الحقيقية تُرفض تلقائياً، حتى لو كانت نتيجة
    // الـ softmax "واثقة" ظاهرياً — لأن ثقة الـ softmax غير موثوقة أصلاً على صور لم
    // يتدرب عليها النموذج بأي شكل مشابه.
    const score = noveltyScore(embedding);
    const noveltyRejected = score !== null && score > NOVELTY_GATE.threshold;
    if (noveltyRejected) {
      predictedClass = "Other";
      // لا نخترع رقم ثقة هنا (كانت نسخة سابقة تعرض 100% وهو مضلل تماماً): القرار مبني على
      // مسافة إحصائية مستقلة عن الـ softmax، مش على "ثقة تصنيف". displayResult لن يعرض أي
      // نسبة مئوية لهذه الحالة تحديداً - راجع noveltyRejected هناك.
    }

    displayResult(predictedClass, confidence, probs, noveltyRejected);
  } catch (err) {
    console.error("Inference failed:", err);
    alert("حدث خطأ أثناء التحليل. حاول مرة أخرى بصورة أخرى.");
  } finally {
    diagnoseBtn.disabled = false;
    diagnoseBtn.textContent = "🔍 تشخيص الصورة";
  }
});

function displayResult(predictedClass, confidence, allProbs, noveltyRejected = false) {
  const resultCard = document.getElementById("result-card");
  const badge = document.getElementById("result-badge");
  const confFill = document.getElementById("confidence-fill");
  const confPct = document.getElementById("confidence-pct");
  const explanationText = document.getElementById("explanation-text");
  const probList = document.getElementById("prob-list");
  const ttaNote = document.getElementById("tta-note");

  const isOther = predictedClass === "Other";
  const isLowConfidence = !isOther && confidence < CONFIDENCE_THRESHOLD;
  const isInconclusive = isOther || isLowConfidence;
  const isHealthy = predictedClass === "Healthy" && !isInconclusive;

  if (isInconclusive) {
    badge.textContent = isOther ? "غير متعلق بالدواجن" : "نتيجة غير واضحة";
    badge.className = "result-badge badge-unclear";
    if (noveltyRejected) {
      explanationText.textContent =
        "هذه الصورة بعيدة إحصائياً عن كل الحالات المعروفة لدى النموذج (بغض النظر عن أي احتمال أظهره). " +
        "على الأرجح لا تحتوي على دجاجة أو فضلات بشكل واضح. يرجى التقاط صورة أقرب وأوضح تتضمن الطائر أو الفضلات مباشرة.";
    } else {
      explanationText.textContent = isOther
        ? EXPLANATION_TEMPLATES.Other
        : EXPLANATION_TEMPLATES.Inconclusive;
    }
  } else {
    badge.textContent = CLASS_NAMES_AR[predictedClass];
    badge.className = "result-badge " + (isHealthy ? "badge-healthy" : "badge-disease");
    explanationText.textContent = EXPLANATION_TEMPLATES[predictedClass];
  }

  if (noveltyRejected) {
    // لا نعرض أي نسبة ثقة هنا: قرار الرفض مبني على مسافة إحصائية مستقلة عن الـ softmax،
    // وعرض رقم "ثقة" بجانبه كان سيكون مضللاً (نسخة سابقة كانت تعرض 100% بالغلط).
    confFill.style.width = "0%";
    confPct.textContent = "قرار مستقل (بدون نسبة ثقة)";
  } else {
    const pct = Math.round(confidence * 100);
    confFill.style.width = pct + "%";
    const { label } = confidenceLabel(confidence);
    confPct.textContent = `${label} (${pct}%)`;
  }

  ttaNote.textContent = NOVELTY_GATE
    ? "↻ النتيجة مبنية على متوسط 3 قراءات للصورة (TTA)، مع بوابة رفض إضافية مستقلة تتحقق من مدى قرب الصورة فعلياً من الحالات المعروفة."
    : "↻ النتيجة مبنية على متوسط 3 قراءات للصورة (Test-Time Augmentation) لتقليل تأثير زاوية التصوير.";

  const sorted = CLASS_NAMES.map((cls, i) => ({ cls, prob: allProbs[i] })).sort((a, b) => b.prob - a.prob);
  probList.innerHTML = sorted
    .map(
      ({ cls, prob }) => `
      <div class="prob-item">
        <span class="name">${CLASS_NAMES_AR[cls].split(" (")[0]}</span>
        <div class="bar-bg"><div class="bar-fill" style="width:${Math.round(prob * 100)}%"></div></div>
        <span class="pct">${Math.round(prob * 100)}%</span>
      </div>`
    )
    .join("");

  resultCard.classList.add("visible");
  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });

  lastResultForFeedback = {
    predictedClass,
    confidence,
    isInconclusive,
    timestamp: new Date().toISOString(),
  };
  resetFeedbackUI();
}

const FEEDBACK_STORAGE_KEY = "poultry_app_feedback_log";
const fbYesBtn = document.getElementById("fb-yes");
const fbNoBtn = document.getElementById("fb-no");
const fbThanks = document.getElementById("feedback-thanks");
const fbExportBtn = document.getElementById("fb-export");

function resetFeedbackUI() {
  fbYesBtn.classList.remove("selected");
  fbNoBtn.classList.remove("selected");
  fbThanks.style.display = "none";
}

function saveFeedback(isCorrect) {
  if (!lastResultForFeedback) return;
  let log = [];
  try {
    log = JSON.parse(localStorage.getItem(FEEDBACK_STORAGE_KEY) || "[]");
  } catch { log = []; }
  log.push({ ...lastResultForFeedback, userSaysCorrect: isCorrect });
  localStorage.setItem(FEEDBACK_STORAGE_KEY, JSON.stringify(log));
  fbThanks.style.display = "block";
}

fbYesBtn.addEventListener("click", () => {
  fbYesBtn.classList.add("selected");
  fbNoBtn.classList.remove("selected");
  saveFeedback(true);
});
fbNoBtn.addEventListener("click", () => {
  fbNoBtn.classList.add("selected");
  fbYesBtn.classList.remove("selected");
  saveFeedback(false);
});
fbExportBtn.addEventListener("click", () => {
  const log = localStorage.getItem(FEEDBACK_STORAGE_KEY) || "[]";
  const blob = new Blob([log], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "poultry_app_feedback.json";
  a.click();
  URL.revokeObjectURL(url);
});

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("service-worker.js").catch((err) => {
      console.warn("Service worker registration failed:", err);
    });
  });
}

let deferredPrompt = null;
const installBanner = document.getElementById("install-banner");
const installBtn = document.getElementById("install-btn");

window.addEventListener("beforeinstallprompt", (e) => {
  e.preventDefault();
  deferredPrompt = e;
  installBanner.classList.add("visible");
});

installBtn.addEventListener("click", async () => {
  if (!deferredPrompt) return;
  deferredPrompt.prompt();
  await deferredPrompt.userChoice;
  deferredPrompt = null;
  installBanner.classList.remove("visible");
});

window.addEventListener("appinstalled", () => {
  installBanner.classList.remove("visible");
});
