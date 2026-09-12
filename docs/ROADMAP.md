# Chessfly Roadmap

## M0 — 可執行 SNN 骨架（已完成）

- [x] 記錄來源、既有結論與科學界線
- [x] 零依賴 LIF 時間步進核心
- [x] 稀疏有向突觸與 delay queue
- [x] population action decoder
- [x] 命令列 demo 和單元測試

完成條件：新環境能以 Python 3.9+ 執行測試與 demo。

## M0.5 — Chess closed loop（已完成）

- [x] 32-channel 行列群體棋步解碼
- [x] python-chess 合法棋步與 PGN
- [x] Stockfish Elo / movetime UCI adapter
- [x] 明確標示的 toy SNN smoke controller
- [x] run manifest、逐步 JSONL 與 PGN
- [x] 科學紀錄片感影片方向與 59 秒 storyboard

完成條件：toy SNN 可與本機 Stockfish 走完整個 smoke match，且每一步能從
artifact 追溯神經輸出、合法棋步選擇與引擎評分。

## M1 — MaleCNS retina 與 reference 子圖（已完成）

- [x] 下載並驗證公開 annotation、neurotransmitter 與 connection-weight Feather
- [x] 重現 166,700 neurons、25,582,938 edges、124,177,617 contacts
- [x] 建立 mapped R1–R6/R8 → descending neuron 的三跳 reference 子圖
- [x] 建立含來源 URL、bytes 與 SHA-256 的本機 provenance manifest
- [x] 產生標準化 graph 與 retina manifests
- [x] 比較 3／4／5／6／8 hops 並記錄實際規模

完成結果：retina 映射 4,146 個 inputs；三跳、contact≥5 子圖包含 8,598
neurons、70,308 edges、1,290,996 contacts、2,147 個 path inputs 與 960 個
path outputs。八跳因膨脹到 160,285 neurons 而不再作為 reference 子圖。

## M2 — MaleCNS chess baseline（已完成）

- [x] 棋盤 RGB → R1–R6 luminance / R8 color 刺激
- [x] descending population → 32-channel chess decoder
- [x] 記錄 spike raster、動作、reward 與 episode seed
- [x] 與隨機、固定及 shuffled-connectome baseline 比較
- [x] 100 個 readout seeds 的重播與分布報告

完成條件：100 個固定 seed 可重播，且報告不只呈現最佳一次結果。

完成結果：10-position depth-10 工程基線與 100-seed readout sweep 均已輸出。
100 個 seed 可精確重播並產生 13 種合法起手，但每個 500 ms decision 只有
2 個 descending spikes；這個稀疏限制保留在報告，不宣稱已展現棋力。

## M3 — 可塑性

- [ ] 先鎖定小範圍 plastic edges
- [ ] 實作 reward-modulated eligibility trace
- [ ] 設定 weight bounds、homeostasis 與 checkpoint
- [ ] 做 no-reward / reversed-reward / shuffled-sign 消融

完成條件：能區分拓撲、初始權重、reward 與隨機波動的效果。

## M4 — Native full graph and video

- [ ] C++17 event-driven backend 與 Python reference parity tests
- [ ] 完整 retained MaleCNS graph 模式
- [ ] 100-FEN benchmark 與 20 場對局 controls
- [x] 以真實 run telemetry 生成 9:16 social cut
- [ ] 生成 16:9 full match

## 暫不進行

- 使用真實資金交易
- 未經限制直接控制實體馬達
- 在沒有 baseline 與消融的情況下宣稱 emergent intelligence、學習或生物等價性
