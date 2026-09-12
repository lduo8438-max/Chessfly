# 對話脈絡與已確認來源

更新日期：2026-09-12

## 核心結論

MaleCNS v1.0 是完整雄性果蠅中央神經系統的 connectome 資料集，而不是可直接執行的「數位大腦」。它提供 neuron、synapse、connection weight、annotation、skeleton、EM volume 與 neurotransmitter prediction 等資料。

Stonkfly 是建立在 MaleCNS v1.0 連線圖上的第三方工程專案。它把 Coinbase 市場價格繪成 RGB 圖表，刺激視覺輸入神經元，再從固定的 descending-neuron readout 解碼 `buy / sell / hold`。它加入候選 KC→MBON 可塑性規則與人工 dopamine reinforcement，但沒有證明能學會獲利。

因此兩者的關係是：

```text
MaleCNS：靜態接線、形態與標註資料
              ↓
神經動力學 + 感覺編碼 + 輸出解碼 + 可塑性（工程選擇）
              ↓
Stonkfly 或 Fly CNS Lab：可執行的實驗系統
```

## 官方與重要連結

- [MaleCNS 官方入口](https://male-cns.janelia.org/)
- [MaleCNS 下載頁](https://male-cns.janelia.org/download/)
- [MaleCNS v1.0 Neuroglancer 場景](https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.json)
- [neuPrint：male-cns:v1.0](https://neuprint.janelia.org/?dataset=male-cns%3Av1.0&qt=findneurons)
- [Male CNS Cell Type Explorer](https://reiserlab.github.io/celltype-explorer-drosophila-male-cns)
- [Cell 論文頁](https://www.cell.com/cell/fulltext/S0092-8674(26)00942-6)
- [Stonkfly GitHub](https://github.com/nftechie/stonkfly)
- [Stonkfly 模型與證據說明](https://github.com/nftechie/stonkfly/blob/main/docs/model.md)

官方專案頁顯示 MaleCNS v1.0 於 2026-06-08 發布，論文於 2026-09-03 發表；專案由 HHMI Janelia FlyEM、University of Cambridge、MRC LMB 與 Google Research 合作完成。

## 資料規模與取得策略

官方下載頁列出的主要檔案包括：

| 資料 | 約略大小 | 初期是否需要 |
|---|---:|---|
| neuron annotations | 13 MB | 是 |
| neuron neurotransmitter predictions | 42 MB | 建議 |
| body statistics | 780 MB | 視需求 |
| complete connection weights | 1.1 GB | 子圖完成後 |
| synaptic partner pairs | 6.8 GB | 否 |
| synapse points | 12.7 GB | 否 |

Chessfly 採用已選定的 public bulk-file 策略；EM volume 與逐突觸座標不屬於
第一階段依賴。

## Chessfly 本機驗證

Chessfly 已下載 v1.0 annotations、neurotransmitters 與 complete connection
weights，並在 `data/manifest.json` 保存實際 bytes 與 SHA-256。原始 weights 表
包含 151,856,684 條 segment-level edges；以 `superclass` 非空的 166,700 個
neuron bodies 同時過濾兩端後，得到 25,582,938 條 directed connections 與
124,177,617 個 synaptic contacts，精確重現 Stonkfly 公開的 retained graph
規模。

## 不能從 connectome 單獨推得的資訊

- 單一通用且正確的神經元動力學參數
- 所有突觸的可靠 excitatory / inhibitory sign 與生理強度
- 從螢幕 pixel 到特定感覺神經元的唯一正確映射
- 哪一組輸出活動應代表某個人工任務動作
- 人工 reward 應如何對應 dopamine activity 與可塑性
- 模擬結果是否代表真實果蠅行為、意識、痛覺或交易能力

所有這些都必須作為顯式假設記錄，並用消融實驗與 baseline 比較。
