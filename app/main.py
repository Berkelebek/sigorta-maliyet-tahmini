from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Sigorta Maliyet Tahmini", version="1.0")


class Sigortali(BaseModel):
    age: int = Field(ge=18, le=64)
    sex: Literal["male", "female"]
    bmi: float = Field(ge=15, le=55)
    children: int = Field(ge=0, le=5)
    smoker: Literal["yes", "no"]
    region: Literal["southwest", "southeast", "northwest", "northeast"]


SAYFA = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sigorta Maliyet Tahmini</title>
<style>
  :root { --bg:#f4f6fb; --kart:#ffffff; --yazi:#1c2333; --soluk:#5d6779; --cizgi:#d9deea; --ana:#2f5bea; --ana-yazi:#ffffff; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#141821; --kart:#1d2330; --yazi:#e8ebf2; --soluk:#9aa3b5; --cizgi:#2d3548; --ana:#5b83ff; --ana-yazi:#0d1220; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--yazi); font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; display:flex; justify-content:center; padding:32px 16px; }
  .kart { background:var(--kart); border:1px solid var(--cizgi); border-radius:16px; padding:28px; width:100%; max-width:460px; }
  h1 { font-size:22px; margin:0 0 4px; }
  p.alt { margin:0 0 22px; color:var(--soluk); font-size:14px; }
  label { display:block; font-size:13px; color:var(--soluk); margin:14px 0 6px; }
  input, select { width:100%; padding:11px 12px; font-size:16px; border:1px solid var(--cizgi); border-radius:10px; background:var(--bg); color:var(--yazi); }
  .iki { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  button { margin-top:22px; width:100%; padding:13px; font-size:16px; font-weight:600; border:0; border-radius:10px; background:var(--ana); color:var(--ana-yazi); cursor:pointer; }
  button:disabled { opacity:.6; cursor:wait; }
  #sonuc { margin-top:22px; padding:18px; border-radius:12px; border:1px solid var(--cizgi); text-align:center; display:none; }
  #sonuc .rakam { font-size:32px; font-weight:700; }
  #sonuc .etiket { font-size:13px; color:var(--soluk); }
  #sonuc.hata { color:#d64545; }
  .not { margin-top:20px; font-size:12px; color:var(--soluk); line-height:1.5; }
</style>
</head>
<body>
<div class="kart">
  <h1>Sigorta Maliyet Tahmini</h1>
  <p class="alt">Bilgileri gir, modelin tahmin ettiği yıllık sağlık sigortası masrafını gör.</p>

  <div class="iki">
    <div><label for="age">Yaş (18-64)</label><input id="age" type="number" min="18" max="64" value="30"></div>
    <div><label for="children">Çocuk sayısı (0-5)</label><input id="children" type="number" min="0" max="5" value="1"></div>
  </div>
  <div class="iki">
    <div><label for="bmi">BMI (15-55)</label><input id="bmi" type="number" step="0.1" min="15" max="55" value="28"></div>
    <div><label for="sex">Cinsiyet</label>
      <select id="sex"><option value="male">Erkek</option><option value="female">Kadın</option></select></div>
  </div>
  <div class="iki">
    <div><label for="smoker">Sigara</label>
      <select id="smoker"><option value="no">İçmiyor</option><option value="yes">İçiyor</option></select></div>
    <div><label for="region">Bölge (ABD)</label>
      <select id="region">
        <option value="northeast">Kuzeydoğu</option><option value="northwest">Kuzeybatı</option>
        <option value="southeast">Güneydoğu</option><option value="southwest">Güneybatı</option>
      </select></div>
  </div>

  <button id="btn" type="button">Tahmin et</button>
  <div id="sonuc"></div>
  <p class="not">Model, 1338 satırlık ABD veri setiyle eğitildi (RandomForest). Eğitim amaçlıdır, gerçek fiyatlama için kullanılamaz.</p>
</div>

<script>
  var btn = document.getElementById("btn");
  var kutu = document.getElementById("sonuc");

  function deger(id) { return document.getElementById(id).value; }

  btn.addEventListener("click", function () {
    var veri = {
      age: parseInt(deger("age"), 10),
      sex: deger("sex"),
      bmi: parseFloat(deger("bmi")),
      children: parseInt(deger("children"), 10),
      smoker: deger("smoker"),
      region: deger("region")
    };
    btn.disabled = true;
    kutu.style.display = "block";
    kutu.className = "";
    kutu.textContent = "Hesaplanıyor...";

    fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(veri)
    })
    .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, j: j }; }); })
    .then(function (s) {
      if (!s.ok) { throw new Error("Girdiğin değerleri kontrol et (yaş, BMI ve çocuk sayısı aralık dışında olabilir)."); }
      var tutar = s.j.predicted_charges.toLocaleString("en-US", { maximumFractionDigits: 0 });
      kutu.innerHTML = '<div class="etiket">Tahmini yıllık masraf</div><div class="rakam">$' + tutar + '</div>';
    })
    .catch(function (e) {
      kutu.className = "hata";
      kutu.textContent = e.message;
    })
    .then(function () { btn.disabled = false; });
  });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def ana_sayfa():
    return SAYFA


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(kisi: Sigortali):
    X = pd.DataFrame([kisi.model_dump()])
    tahmin = float(model.predict(X)[0])
    return {"predicted_charges": round(tahmin, 2)}
