// 全局變數
let labels = [];
let samples = {};           // { label: [landmarkVectors...] }
let model = null;
let prediction = null;
let gestureStart = null;
let currentGesture = null;
let recognizing = false;
let isTraining = false;
let lastLandmarks = null;
let isRecording = false;
let currentRecordingLabel = null;
let recordingInterval = null;

// DOM元素
const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const predictionOverlay = document.getElementById('prediction-overlay');
const cameraStatus = document.getElementById('camera-status');
const trainingStatus = document.getElementById('training-status');
const progressBar = document.getElementById('progress');
const epochInfo = document.getElementById('epoch-info');
const toast = document.getElementById('toast');

// 啟動
window.addEventListener('DOMContentLoaded', init);

// 初始化攝影機
async function initCamera() {
  try {
    showToast('正在啟動相機...');
    // 設定相機尺寸，保持適中的分辨率
    const stream = await navigator.mediaDevices.getUserMedia({ 
      video: { 
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: 'user'
      } 
    });
    video.srcObject = stream;
    
    // 移除視頻的鏡像效果，我們將在繪製骨架時手動處理座標
    video.style.transform = ''; 
    
    return new Promise(resolve => {
      video.onloadedmetadata = () => {
        console.log(`視頻尺寸: ${video.videoWidth} x ${video.videoHeight}`);
        cameraStatus.textContent = '相機已就緒';
        setTimeout(() => {
          cameraStatus.style.opacity = '0';
        }, 1000);
        resolve();
      };
    });
  } catch (error) {
    console.error('無法啟動相機:', error);
    cameraStatus.textContent = '相機啟動失敗';
    cameraStatus.classList.add('error');
    throw error;
  }
}

// 初始化 MediaPipe Hands
const hands = new Hands({ 
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}` 
});
hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.7
});
hands.onResults(onHandsResults);

// 初始化
async function init() {
  try {
    setupEventListeners();
    await initCamera();
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    drawLoop();
    loadFromLocalStorage();
    showToast('系統已準備就緒！');
  } catch (error) {
    console.error('初始化失敗:', error);
    showToast('初始化失敗，請重新整理頁面', 'error');
  }
}

// 設置所有事件監聽器
function setupEventListeners() {
  // 新增標籤
  document.getElementById('add-label').addEventListener('click', addNewLabel);

  // 訓練按鈕
  document.getElementById('train').addEventListener('click', trainModel);

  // 儲存模型
  document.getElementById('save-model').addEventListener('click', saveModel);

  // 載入模型
  document.getElementById('load-model').addEventListener('change', loadModel);

  // 啟動辨識
  document.getElementById('start-recognize').addEventListener('click', toggleRecognition);

  // 清除輸出
  document.getElementById('clear-output').addEventListener('click', () => {
    document.getElementById('text-output').value = '';
  });

  // 複製輸出
  document.getElementById('copy-output').addEventListener('click', () => {
    const textarea = document.getElementById('text-output');
    textarea.select();
    document.execCommand('copy');
    showToast('文字已複製到剪貼簿');
  });
}

// 調整畫布尺寸
function resizeCanvas() {
  // 確保畫布尺寸與視頻實際尺寸一致
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  
  // 確保畫布位置與視頻一致
  canvas.style.width = video.offsetWidth + 'px';
  canvas.style.height = video.offsetHeight + 'px';
  
  // 確保預測覆蓋層的尺寸正確
  if (predictionOverlay) {
    predictionOverlay.style.fontSize = (video.offsetHeight * 0.05) + 'px';
  }
}

// 每幀送入 MediaPipe
async function drawLoop() {
  try {
    // 不使用 flipHorizontal 選項，因為我們會手動翻轉座標
    await hands.send({ 
      image: video
    });
  } catch (error) {
    console.error('處理影像錯誤:', error);
  }
  requestAnimationFrame(drawLoop);
}

// 處理手部偵測結果
function onHandsResults(results) {
  // 確保畫布大小與視頻一致
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
  }
  
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const landmarks = results.multiHandLandmarks[0];
    
    // 手動翻轉座標系統以正確顯示手部
    const flippedLandmarks = landmarks.map(point => {
      return {
        x: point.x,  // 水平翻轉 x 座標
        y: point.y,
        z: point.z
      };
    });
    
    // 繪製手部骨架
    drawConnectors(ctx, flippedLandmarks, HAND_CONNECTIONS, {
      color: '#00FF00',
      lineWidth: 3
    });
    drawLandmarks(ctx, flippedLandmarks, {
      color: '#FF0000',
      lineWidth: 1,
      radius: 3
    });

    // 使用翻轉後的座標進行處理
    handleFrame(flippedLandmarks);
    
    // 如果正在錄製，顯示提示
    if (isRecording) {
      ctx.save();
      ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
      ctx.beginPath();
      ctx.arc(canvas.width / 2, canvas.height / 2, 20, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  } else {
    if (recognizing) {
      predictionOverlay.textContent = '沒有偵測到手';
      predictionOverlay.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
    }
    resetGesture();
  }

  // 如果正在錄製但沒偵測到手，顯示警告
  if (isRecording && (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0)) {
    showToast('無法偵測到手，請將手放在鏡頭前', 'warning');
  }
}

// 將 21 個關鍵點轉成一維向量
function landmarksToVector(landmarks) {
  // 確保座標一致性
  const normalizedLandmarks = landmarks.map(pt => {
    // 如果已經是翻轉過的座標，就不需要再翻轉
    if (pt.hasOwnProperty('originalX')) {
      return pt;
    } else {
      // 確保轉換一致性
      return {
        x: pt.x,
        y: pt.y,
        z: pt.z
      };
    }
  });
  return normalizedLandmarks.flatMap(pt => [pt.x, pt.y, pt.z]);
}

// 新增標籤
function addNewLabel() {
  const input = document.getElementById('new-label');
  const name = input.value.trim();
  
  if (!name) {
    showToast('請輸入標籤名稱', 'error');
    return;
  }
  
  if (labels.includes(name)) {
    showToast('標籤已存在', 'error');
    return;
  }
  
  labels.push(name);
  samples[name] = [];
  renderLabels();
  input.value = '';
  showToast(`已新增「${name}」標籤`);
  saveToLocalStorage();
}

// 渲染標籤列表
function renderLabels() {
  const container = document.getElementById('labels');
  container.innerHTML = '';
  
  if (labels.length === 0) {
    container.innerHTML = '<div class="empty-state">尚未新增任何標籤</div>';
    return;
  }
  
  labels.forEach((label, index) => {
    const div = document.createElement('div');
    div.className = 'label-item';
    
    const labelText = document.createElement('div');
    labelText.className = 'label-text';
    labelText.textContent = label;
    div.appendChild(labelText);
    
    const controls = document.createElement('div');
    controls.className = 'label-controls';
    
    const sampleCount = document.createElement('span');
    sampleCount.className = 'sample-count';
    sampleCount.textContent = samples[label].length;
    controls.appendChild(sampleCount);
    
    const recordBtn = document.createElement('button');
    recordBtn.innerHTML = '<span class="material-icons">fiber_manual_record</span>';
    recordBtn.title = '錄製樣本';
    recordBtn.onclick = () => toggleRecording(label);
    controls.appendChild(recordBtn);
    
    const deleteBtn = document.createElement('button');
    deleteBtn.innerHTML = '<span class="material-icons">delete</span>';
    deleteBtn.title = '刪除標籤';
    deleteBtn.style.backgroundColor = '#ea4335';
    deleteBtn.onclick = () => deleteLabel(index);
    controls.appendChild(deleteBtn);
    
    div.appendChild(controls);
    container.appendChild(div);
  });
  
  updateCollectionUI();
}

// 更新資料收集UI
function updateCollectionUI() {
  const container = document.getElementById('collection');
  container.innerHTML = '';
  
  if (labels.length === 0) {
    container.innerHTML = '<div class="empty-state">請先新增標籤</div>';
    return;
  }
  
  labels.forEach(label => {
    const progress = document.createElement('div');
    progress.className = 'collection-progress';
    
    const labelDiv = document.createElement('div');
    labelDiv.className = 'collection-label';
    labelDiv.textContent = label;
    progress.appendChild(labelDiv);
    
    const bar = document.createElement('div');
    bar.className = 'progress-bar';
    
    const fill = document.createElement('div');
    fill.className = 'progress';
    const sampleCount = samples[label].length;
    const percentage = Math.min(sampleCount / 30 * 100, 100);
    fill.style.width = `${percentage}%`;
    bar.appendChild(fill);
    
    progress.appendChild(bar);
    
    const count = document.createElement('div');
    count.className = 'collection-count';
    count.textContent = `${sampleCount} 樣本`;
    progress.appendChild(count);
    
    container.appendChild(progress);
  });
}

// 切換錄製狀態
function toggleRecording(label) {
  if (isRecording && currentRecordingLabel === label) {
    // 停止錄製
    stopRecording();
  } else {
    // 開始錄製
    startRecording(label);
  }
}

// 開始錄製
function startRecording(label) {
  if (isRecording) {
    stopRecording();
  }
  
  isRecording = true;
  currentRecordingLabel = label;
  showToast(`正在錄製「${label}」樣本`);
  
  // 每0.3秒自動收集一個樣本
  recordingInterval = setInterval(() => {
    if (lastLandmarks) {
      samples[label].push(landmarksToVector(lastLandmarks));
      renderLabels();
      showToast(`已收集「${label}」樣本: ${samples[label].length}`, 'info', 500);
    }
  }, 100);
  
  // 5秒後自動停止
  setTimeout(stopRecording, 5000);
}

// 停止錄製
function stopRecording() {
  if (!isRecording) return;
  
  clearInterval(recordingInterval);
  isRecording = false;
  showToast(`已完成「${currentRecordingLabel}」樣本錄製`);
  currentRecordingLabel = null;
  saveToLocalStorage();
}

// 刪除標籤
function deleteLabel(index) {
  if (confirm(`確定要刪除「${labels[index]}」標籤和所有相關樣本嗎？`)) {
    const label = labels[index];
    delete samples[label];
    labels.splice(index, 1);
    renderLabels();
    showToast(`已刪除「${label}」標籤`);
    saveToLocalStorage();
  }
}

// 每幀呼叫，更新 lastLandmarks 且處理辨識或收集
function handleFrame(landmarks) {
  lastLandmarks = landmarks;
  
  if (model && recognizing) {
    try {
      const input = tf.tensor([landmarksToVector(landmarks)]);
      const probs = model.predict(input);
      const scores = probs.arraySync()[0];
      const maxIdx = scores.indexOf(Math.max(...scores));
      const pred = labels[maxIdx];
      const confidence = scores[maxIdx];
      
      updatePredictionDisplay(pred, confidence);
      recognizeGesture(pred);
      
      input.dispose();
      probs.dispose();
    } catch (error) {
      console.error('預測錯誤:', error);
    }
  }
}

// 更新預測顯示
function updatePredictionDisplay(prediction, confidence) {
  predictionOverlay.style.display = 'block';
  predictionOverlay.textContent = `${prediction} (${(confidence * 100).toFixed(1)}%)`;
  predictionOverlay.style.backgroundColor = `rgba(66, 133, 244, ${confidence * 0.8 + 0.2})`;
}

// 模型訓練
async function trainModel() {
  // 檢查是否有足夠的標籤和樣本
  if (labels.length < 2) {
    return showToast('至少需要兩個標籤才能訓練模型', 'error');
  }
  
  let totalSamples = 0;
  for (const label of labels) {
    if (samples[label].length < 10) {
      return showToast(`「${label}」至少需要10個樣本（目前${samples[label].length}個）`, 'error');
    }
    totalSamples += samples[label].length;
  }
  
  if (isTraining) {
    return showToast('模型訓練中，請稍候', 'warning');
  }
  
  try {
    isTraining = true;
    showToast('開始訓練模型...');
    trainingStatus.classList.remove('hidden');
    
    // 準備訓練資料
    const allSamples = [];
    const allLabels = [];
    labels.forEach((lbl, i) => {
      samples[lbl].forEach(vec => {
        allSamples.push(vec);
        allLabels.push(i);
      });
    });
    
    // 建立並編譯模型
    const xs = tf.tensor2d(allSamples);
    const ys = tf.oneHot(tf.tensor1d(allLabels, 'int32'), labels.length);
    
    model = tf.sequential();
    model.add(tf.layers.dense({inputShape: [63], units: 128, activation: 'relu'}));
    model.add(tf.layers.dropout({rate: 0.2}));
    model.add(tf.layers.dense({units: 64, activation: 'relu'}));
    model.add(tf.layers.dropout({rate: 0.2}));
    model.add(tf.layers.dense({units: 64, activation: 'relu'}));
    model.add(tf.layers.dense({units: labels.length, activation: 'softmax'}));
    
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
    
    // 訓練模型並顯示進度
    const epochs = 50;
    await model.fit(xs, ys, {
      epochs: epochs,
      batchSize: 16,
      shuffle: true,
      callbacks: {
        onEpochBegin: (epoch) => {
          epochInfo.textContent = `Epoch: ${epoch+1}/${epochs}`;
        },
        onEpochEnd: (epoch, logs) => {
          const accuracy = logs.acc;
          progressBar.style.width = `${((epoch+1) / epochs) * 100}%`;
          epochInfo.textContent = `Epoch: ${epoch+1}/${epochs} (準確率: ${(accuracy * 100).toFixed(1)}%)`;
          console.log(`Epoch ${epoch+1}: 準確率 = ${(accuracy * 100).toFixed(1)}%`);
        }
      }
    });
    
    xs.dispose();
    ys.dispose();
    
    showToast('模型訓練完成！', 'success');
    saveToLocalStorage();
  } catch (error) {
    console.error('訓練失敗:', error);
    showToast('模型訓練失敗', 'error');
  } finally {
    isTraining = false;
    setTimeout(() => {
      trainingStatus.classList.add('hidden');
    }, 1000);
  }
}

// 儲存模型
async function saveModel() {
  if (!model) {
    return showToast('尚未有模型可儲存', 'error');
  }
  
  try {
    showToast('正在儲存模型...');
    await model.save('downloads://hand_gesture_model');
    showToast('模型儲存完成', 'success');
  } catch (error) {
    console.error('儲存模型失敗:', error);
    showToast('儲存模型失敗', 'error');
  }
}

// 載入模型
async function loadModel(e) {
  const file = e.target.files[0];
  if (!file) return;
  
  try {
    showToast('正在載入模型...');
    model = await tf.loadLayersModel(tf.io.browserFiles([file]));
    showToast('模型載入完成', 'success');
  } catch (error) {
    console.error('載入模型失敗:', error);
    showToast('載入模型失敗', 'error');
  }
}

// 切換辨識狀態
function toggleRecognition() {
  const btn = document.getElementById('start-recognize');
  
  if (recognizing) {
    // 停止辨識
    recognizing = false;
    btn.innerHTML = '<span class="material-icons">play_circle</span> 啟動辨識';
    predictionOverlay.style.display = 'none';
    showToast('辨識已停止');
  } else {
    // 檢查模型
    if (!model) {
      return showToast('請先訓練或載入模型', 'error');
    }
    
    // 啟動辨識
    recognizing = true;
    btn.innerHTML = '<span class="material-icons">stop_circle</span> 停止辨識';
    gestureStart = null;
    showToast('辨識已啟動');
  }
}

// 重置手勢狀態
function resetGesture() {
  gestureStart = null;
  currentGesture = null;
}

// 手勢持續判定與文字輸出
function recognizeGesture(pred) {
  const threshold = parseFloat(document.getElementById('hold-threshold').value) * 1000;
  const now = Date.now();
  
  if (pred !== currentGesture) {
    currentGesture = pred;
    gestureStart = now;
  } else if (gestureStart && now - gestureStart >= threshold) {
    appendText(pred);
    gestureStart = null;
    showToast(`已辨識: ${pred}`, 'success', 800);
  }
}

// 文字輸出
function appendText(text) {
  const ta = document.getElementById('text-output');
  ta.value = ta.value + text + ' ';
  ta.focus();
  ta.selectionStart = ta.selectionEnd = ta.value.length;
}

// 儲存到本地儲存
function saveToLocalStorage() {
  try {
    const data = {
      labels: labels,
      samples: samples
    };
    localStorage.setItem('handGestureData', JSON.stringify(data));
  } catch (error) {
    console.error('儲存本地資料失敗:', error);
  }
}

// 從本地儲存載入
function loadFromLocalStorage() {
  try {
    const dataStr = localStorage.getItem('handGestureData');
    if (dataStr) {
      const data = JSON.parse(dataStr);
      labels = data.labels || [];
      samples = data.samples || {};
      renderLabels();
      showToast('從本地儲存載入資料');
    }
  } catch (error) {
    console.error('載入本地資料失敗:', error);
  }
}

// 顯示提示訊息
function showToast(message, type = 'info', duration = 2000) {
  const toastEl = document.getElementById('toast');
  toastEl.textContent = message;
  
  // 設置顏色
  if (type === 'error') {
    toastEl.style.backgroundColor = '#ea4335';
  } else if (type === 'success') {
    toastEl.style.backgroundColor = '#34a853';
  } else if (type === 'warning') {
    toastEl.style.backgroundColor = '#fbbc04';
    toastEl.style.color = '#000';
  } else {
    toastEl.style.backgroundColor = '#4285f4';
  }
  
  // 顯示
  toastEl.style.opacity = '1';
  
  // 隱藏
  setTimeout(() => {
    toastEl.style.opacity = '0';
  }, duration);
}
