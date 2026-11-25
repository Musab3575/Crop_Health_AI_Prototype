# Crop Health AI

## Run locally

```powershell
cd d:\Crop_Health_AI_Prototype
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd d:\Crop_Health_AI_Prototype\frontend
npm install
npm run dev
```

Open http://127.0.0.1:3000. API: http://127.0.0.1:8000/docs.

## Model

Actual model: ml_assets/model/mobilenetv2_stage2_best.keras

Input: 224 x 224 RGB. Preprocessing: tf.keras.applications.mobilenet_v2.preprocess_input. Mapping: ml_assets/metadata/class_mapping.json.

This prototype uses a held-out research test result of 95.10% accuracy on 10,704 images. Field performance is not guaranteed.

## Camera and model confidence

The scanner requests browser camera permission only after you select **Camera**. Camera access works on secure origins, including `http://localhost` and `http://127.0.0.1`. In Chrome, allow Camera for the site and check Windows **Settings → Privacy & security → Camera**. Close other apps using the camera if Chrome reports that it is busy. The scanner can retry and select another camera when multiple devices are available.

The model's final layer is a 38-class softmax. Its outputs already sum to one, so the API reports those output values directly as model confidence; it does not apply softmax a second time. Scores are not calibrated guarantees of correctness. A configurable 0.40 threshold marks relatively low scores as uncertain; this display threshold has not been scientifically validated. Set `LOW_CONFIDENCE_THRESHOLD` in `backend/.env` if you want to configure it. Existing history entries retain the scores recorded when they were created.
