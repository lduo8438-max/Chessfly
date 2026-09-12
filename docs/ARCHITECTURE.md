# Chessfly 技術架構

## 設計原則

1. **資料與動力學分離**：MaleCNS loader 只輸出標準化節點與邊；SNN engine 不知道 neuPrint。
2. **控制器與環境分離**：遊戲、模擬機器人與其他環境共用 observation/action 介面。
3. **所有生物學假設可追蹤**：突觸 sign、weight scaling、delay、cell selection 與 readout 都寫入具版本的 experiment config。
4. **先子圖後全圖**：先建立可以測試和解釋的 end-to-end 閉環，再擴大規模。
5. **paper/simulation first**：不把研究中的神經輸出直接接到資金或實體致動器。

## 模組邊界

```text
MaleCNS / neuPrint
        │
        ▼
  data adapter ──► normalized graph ──► SNN engine
                                            │ spikes
environment ──observation──► encoder        ▼
     ▲                                  decoder
     └──────────── action ◄─────────────────┘
                         │
                         ▼
                recorder / evaluator
```

目前已建立 `SNN reference engine`、32-channel chess decoder、toy chess brain、
Stockfish adapter 與 run artifact recorder。toy brain 只用於打通真實棋規、UCI、
神經 spike 與記錄路徑，所有輸出都標記為 `toy-not-male-cns`。

## Chess move interface

每個局面只執行一次神經時間窗。Descending neurons 以穩定 hash 均衡分配到
32 個 population：起點 file 8 組、起點 rank 8 組、終點 file 8 組、終點
rank 8 組。每個合法棋步的分數是四個對應 population normalized firing rate
的總和；最高分勝出。合法棋步遮罩只排除不合法輸出，不注入 Stockfish 建議。

## 標準化 graph schema

節點至少包含：

- `body_id`: MaleCNS segment/body ID
- `cell_type`, `class`, `side`: 原始 annotation
- `neurotransmitter`: 預測結果與 confidence
- `role`: input / hidden / output（實驗指定，不是原始生物學事實）

邊至少包含：

- `pre_body_id`, `post_body_id`
- `synapse_count`: 原始 connection strength
- `sign`: 來源與推定方法必須記錄
- `sim_weight`: 經縮放後的模擬權重
- `delay_steps`: 工程假設或由形態估算

## 可重現性要求

每次實驗輸出一份 manifest：資料集版本、query、節點/邊數、程式 commit、參數、random seed、環境版本與評估結果。對每個新模型至少保留三個 baseline：隨機動作、固定動作、相同拓撲但打亂權重或標籤。
