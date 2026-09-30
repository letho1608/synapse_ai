# Công trình liên quan: các paper và dự án tương tự đã làm thế nào

*Khảo sát ngày 2026-09-30. Mục đích: đặt Synapse cạnh các hệ thống chạy LLM phân tán trên máy phổ thông, rút ra những gì nên học và những gì README đang nói quá.*

## 1. Kết luận ngắn

1. **Không ai tăng tốc được một request đơn lẻ bằng pipeline parallelism thuần.**
   - exo tự đo: Llama 3.2 3B chạy 49.3 tok/s trên 1 máy, nhưng chỉ còn 39.7 tok/s trên 3 máy.
   - llama.cpp RPC ghi rõ nó chỉ để gom bộ nhớ, không để chạy nhanh hơn.
   - Paper nào cho request đơn nhanh hơn đều phải thêm một cơ chế khác:
     - speculative decoding chạy trong pipeline (PipeInfer, FlowSpec);
     - tensor parallel (distributed-llama, TPI-LLM, Galaxy);
     - hoặc so với baseline không chứa nổi model, phải offload xuống đĩa (Petals, prima.cpp).
2. **Giá trị thật của pipeline parallelism là dung lượng và thông lượng.**
   - Dung lượng: chạy được model mà một máy không chứa nổi.
   - Thông lượng: tổng token/s tăng khi có nhiều request cùng lúc (exo: 3 máy xử lý 108.8 tok/s so với 49.3).
   - Synapse đang đúng hướng này. Phần "PIPELINE BUBBLE-FREE / 333 tokens/sec" trong README thì mâu thuẫn với mọi bằng chứng trên khi áp cho decode một request.
3. **Nút cổ chai là độ trễ mạng mỗi hop, không phải băng thông.** exo, TPI-LLM và Petals (bảng 3) đều thấy vậy. Muốn giảm độ trễ cho một token, cách hiệu quả nhất là giảm số stage: lượng tử hoá trọng số để ít máy hơn chứa được model. Nén activation có ích nhưng là thứ yếu.
4. **Các phần Synapse đã ngang hoặc hơn tài liệu:**
   - Codec INT8 per-row + outlier cho hidden state. Petals chỉ dùng blockwise 8-bit và báo "giảm một nửa băng thông". Synapse nhỏ hơn fp32 3.6×, KL ≈ 5e-5, có test chặn chất lượng.
   - Mỗi request mang `partition_plan` riêng.
   - Kiểm tra parity với HF reference.
5. **Các phần Synapse còn thiếu so với tài liệu, theo thứ tự đáng làm:**
   - (a) Batching nhiều request.
   - (b) Speculative decoding verify trong một lượt pipeline.
   - (c) Mô hình chi phí đo thật cho partition (profiling + DP/ILP), thay cho trọng số heuristic.
   - (d) Bản sao dự phòng của layer, để phục hồi giữa chừng như Petals.
   - (e) Lượng tử hoá trọng số 4/8-bit để giảm số stage.
   - (f) Benchmark thật trên nhiều máy, có giả lập mạng.

## 2. Bảng so sánh

| Hệ thống | Kiểu song song | Chia layer thế nào | Truyền giữa máy | Chịu lỗi | Kết quả chính |
|---|---|---|---|---|---|
| **exo** (exo-explore) | PP, về sau thêm TP (MLX distributed) | Ring, tỉ lệ theo RAM (`RingMemoryWeightedPartitioningStrategy`); bản mới "topology-aware" | gRPC / MLX; RDMA qua Thunderbolt 5 | Tự phát hiện máy, không phục hồi giữa request | 1 request chậm đi khi thêm máy; nhiều request thì 2.2× thông lượng với 3 máy; TP 1.8× (2 máy), 3.2× (4 máy) trên Mac RDMA |
| **Petals** (ACL demo 2023, NeurIPS 2023) | PP qua Internet | Mỗi server tự chọn đoạn layer liên tục đang có thông lượng thấp nhất; tái cân bằng khi lợi ≥ 20% | Hidden state nén dynamic blockwise 8-bit; trọng số LLM.int8 | Client giữ input đã gửi cho từng stage; server chết thì phát lại cho server thay thế, chi phí O(t) | BLOOM-176B ~1 step/s; Llama 2 70B 2.29 step/s ở 1 Gbit/s, gấp ~16.5× offloading |
| **distributed-llama** (b4rtaz) | TP, root + workers | Chia đều; số node phải là 2^n và ≤ số KV head | Đồng bộ bằng Q80 | Không | Chạy được 70B trên 4 Mac mini; nhạy với độ trễ all-reduce |
| **llama.cpp RPC** | Chia layer theo thiết bị ggml | Tỉ lệ bộ nhớ trống | Serialize từng op ggml qua TCP | Không | Chỉ gom bộ nhớ, không nhanh hơn |
| **prima.cpp** (arXiv 2504.08791) | Piped-ring: PP vòng, nhiều vòng cho mỗi token | **Halda**: mô hình hoá độ trễ tính toán / bộ nhớ / đĩa / mạng; duyệt ước số k rồi giải ILP bằng HiGHS | Hidden state F32 (4·e byte mỗi cửa sổ) | Không | 70B Q4K ~600 ms/token trên 4 thiết bị (Mac, laptop, desktop, điện thoại), áp lực bộ nhớ < 6%; exo/dllama OOM trên cùng cụm |
| **TPI-LLM** (arXiv 2410.00531) | TP trên edge | Chia đều + sliding-window nạp trọng số từ đĩa | Star all-reduce (vì độ trễ link là nút cổ chai) | Không | 70B với 3.1 GB mỗi máy; giảm > 80% độ trễ so với Accelerate |
| **Galaxy** (INFOCOM 2024) | TP (attention/MLP) + SP xen kẽ | Theo năng lực thiết bị | Chồng lấp truyền–tính theo tile | Không | Tới 2.5× end-to-end trên edge |
| **EdgeShard** (arXiv 2405.14371) | PP edge + cloud | **Quy hoạch động** chọn thiết bị và điểm cắt, tối ưu độ trễ hoặc thông lượng | — | Không | Giảm 50% độ trễ, 2× thông lượng so với baseline |
| **Helix** (ASPLOS 2025) | PP trên GPU và mạng không đồng nhất | Bài toán **max-flow**, giải bằng MILP: đặt model và định tuyến request cùng lúc | — | — | Tới 3.3× thông lượng, 24–42 node |
| **HexGen** (ICML 2024) | TP + PP bất đối xứng | Thuật toán tiến hoá tìm nhóm song song | — | — | Độ trễ thấp hơn Petals tới 3.5× |
| **Parallax** (arXiv 2509.26182) | PP phi tập trung | Hai pha: đặt layer của nhiều bản sao, rồi chọn đường đi GPU cho từng request | — | Chọn lại đường khi máy thay đổi | Tốt hơn các baseline phi tập trung (abstract không nêu số) |
| **PipeInfer** (SC24), **FlowSpec** (2025), SpecPipe | Speculative decoding trên PP | — | — | — | PipeInfer tới 2.15× so với speculative thường; FlowSpec 1.35–1.77× so với PP trên 5 Jetson Orin Nano |

## 3. Chi tiết từng nhóm và bài học cho Synapse

### 3.1 exo: gần Synapse nhất về trải nghiệm (cụm gia đình, tự phát hiện)
- **Cách làm:**
  - Pipeline theo vòng; mỗi máy nhận số layer tỉ lệ với RAM của nó.
  - Chính exo từng treo bounty vì cách chia này quá đơn giản.
  - Bản mới chuyển sang MLX, có tensor parallel và RDMA Thunderbolt 5 (giảm 99% độ trễ giữa máy).
- **Bài học:**
  - exo công bố benchmark trung thực: một request chậm đi khi thêm máy. Synapse nên trình bày y như vậy.
  - Để có tăng tốc thật, exo phải chuyển sang TP trên đường truyền độ trễ cực thấp. Với LAN/Wi-Fi thường, TP không đáng làm.

### 3.2 Petals: chuẩn mực cho chịu lỗi và cân bằng tải
- **Cân bằng tải:**
  - Mỗi server tự chọn đoạn block liên tục nơi thông lượng tổng đang thấp nhất.
  - Server công bố đoạn đó và thông lượng đo được lên DHT.
  - Định kỳ, server kiểm tra xem đổi đoạn có lợi ≥ 20% không (ngưỡng này để tránh dao động).
- **Định tuyến:**
  - Client đo độ trễ tới các server, rồi tìm đường ngắn nhất qua đồ thị block (beam search / D* Lite).
  - Mỗi đoạn layer có nhiều server cùng giữ.
- **Chịu lỗi giữa lúc sinh token:**
  - Server giữ KV cache.
  - Client giữ input đã gửi cho từng stage.
  - Khi server chết, client phát lại các input đó cho server thay thế, và server thay thế dựng lại KV.
- **So với Synapse:**
  - Đường phân tán của Synapse gửi toàn bộ ngữ cảnh mỗi bước và dùng lại KV khi input là phần nối dài chính xác. Vì vậy, về nguyên tắc, một stage thay thế sẽ tự dựng lại KV ở bước kế tiếp; đó là tương đương cơ chế phát lại của Petals.
  - Chỗ thiếu thật là **bản sao**: mỗi đoạn layer của Synapse chỉ có một chủ. Khi chủ chết, máy khác phải nạp layer (hoặc khôi phục từ file checkpoint trọng số, nhưng file đó nằm trên máy đã chết).
  - Petals giải quyết bằng cách cho nhiều server giữ cùng một đoạn.
- **Nén:**
  - Petals nén hidden state bằng dynamic blockwise 8-bit ("giảm một nửa băng thông, chất lượng không đổi đáng kể").
  - Trọng số dùng LLM.int8 (0.1% outlier giữ 16-bit), nên số node cần cho BLOOM giảm từ 44 xuống 22: độ trễ giảm một nửa và xác suất lỗi cũng giảm.
  - Codec của Synapse cùng ý tưởng outlier nhưng áp cho activation.
- **Cách đánh giá:**
  - 3 server A100 thật; 12 server ảo; 14 server thật ở châu Âu và Bắc Mỹ.
  - Mạng giả lập bằng `tc qdisc`, ba mức: 1 Gbit/s <5 ms, 100 Mbit/s <5 ms, 100 Mbit/s 100 ms.
  - So với offloading.
  - Synapse nên theo khung đánh giá này (xem mục 5).

### 3.3 prima.cpp: chia layer bằng mô hình chi phí đo thật
- Halda không gán trọng số tay. Nó profile từng thiết bị theo từng định dạng lượng tử và backend: FLOPS, băng thông bộ nhớ, tốc độ đọc đĩa, mạng. Từ đó nó viết độ trễ một token thành hàm của số layer mỗi máy, rồi giải bằng ILP.
- Piped-ring cho phép thiết bị yếu RAM giữ ít layer mỗi vòng; mmap + prefetch che thời gian đọc đĩa.
- **So với Synapse:**
  - LACP của Synapse cộng các tài nguyên với trọng số cố định (VRAM, TFLOPS, CPU, RAM, disk, bandwidth, power). Đó là heuristic, không phải mô hình độ trễ.
  - Nên chuyển sang: đo thời gian forward một layer trên mỗi máy (một lần khi join) và đo RTT từng cặp máy. Sau đó cực tiểu `max_i(stage_i) + Σ hop` (tối ưu thông lượng) hoặc `Σ stage + Σ hop` (tối ưu độ trễ).
  - Với chuỗi layer liên tục, quy hoạch động kiểu EdgeShard giải chính xác trong thời gian đa thức, không cần PuLP.

### 3.4 Nhóm tensor parallel trên edge (distributed-llama, TPI-LLM, Galaxy)
- **TP chia nhỏ mỗi layer**, nên về lý thuyết là cách duy nhất rút ngắn thời gian của một token.
- **Cái giá là all-reduce ở mỗi layer.** TPI-LLM chỉ ra độ trễ link mới là thứ quyết định, nên họ dùng star all-reduce. distributed-llama phải ép số node là 2^n và đồng bộ bằng Q80.
- **Kết luận cho Synapse:** với LAN/Wi-Fi độ trễ ~1–10 ms mà model có 30+ layer, TP sẽ tốn hàng chục đến hàng trăm ms cho mỗi token chỉ để đồng bộ. Chỉ nên cân nhắc TP trong một máy nhiều GPU, hoặc giữa các máy nối link độ trễ dưới 100 µs. Giữ PP làm hướng chính là đúng.

### 3.5 Speculative decoding trên pipeline (PipeInfer, FlowSpec, SpecPipe)
- **Vấn đề chung:** decode tuần tự để hầu hết stage ngồi chờ.
- **Lời giải:**
  - Một draft model nhỏ đề xuất nhiều token (dạng cây).
  - Các nhánh draft được đẩy vào pipeline để lấp chỗ trống; stage cuối verify.
  - Nhánh sai bị huỷ sớm ("early inference cancellation").
  - FlowSpec sắp nhánh theo điểm tích luỹ, để token có xác suất cao được verify và chấp nhận sớm hơn.
- **So với Synapse:**
  - Synapse đang dùng đoán n-gram lookahead và bắn `max(1, len(guesses))` micro-batch, mỗi micro-batch mang toàn bộ ngữ cảnh.
  - Đây là dạng yếu của cùng ý tưởng: không có draft model; mỗi đoán là một lượt forward riêng thay vì verify k token trong một lượt; không có huỷ sớm.
  - Việc nên làm tiếp: verify k token đoán trong **một** forward (attention mask cho k vị trí cuối) và huỷ các micro-batch đã lỗi thời. Đây chính là mục 2 trong kế hoạch "nhanh hơn thật" đã đề xuất trước đó.

### 3.6 Lập lịch cho nhiều request (Helix, Parallax, HexGen)
- Cả ba đều tối ưu **thông lượng phục vụ** trên cụm GPU không đồng nhất:
  - Helix giải max-flow bằng MILP.
  - Parallax đặt layer của nhiều bản sao, rồi chọn đường đi cho từng request.
  - HexGen dùng TP/PP bất đối xứng.
- Điểm chung Synapse chưa có:
  - nhiều bản sao cho mỗi đoạn layer;
  - định tuyến theo từng request qua các bản sao;
  - batching nhiều request trong cùng một forward.
- Mỗi request của Synapse đã mang `partition_plan` riêng, nên nền tảng để chọn đường cho từng request (kiểu Parallax) đã có sẵn.

## 4. Đối chiếu Synapse

| Hạng mục | Tài liệu làm gì | Synapse hiện có | Đánh giá |
|---|---|---|---|
| Nén hidden state | Petals: 8-bit blockwise, ~2× | INT8 per-row + 8 outlier/hàng, 3.6× so với fp32, top-1 100%, KL ≈ 5e-5, test chặn | **Ngang/hơn**; có thể dùng làm đóng góp riêng của paper |
| Lượng tử trọng số | Petals LLM.int8; prima.cpp Q4K; dllama Q40 | bnb được phát hiện nhưng chưa được kiểm chứng/cứng hoá | **Thiếu**: đây là cách rẻ nhất để giảm số hop |
| Chia layer | Halda (profiling + ILP), EdgeShard (DP), Helix (max-flow) | LRM/ILP trên điểm tài nguyên có trọng số cố định | **Yếu**: nên đổi sang mô hình độ trễ đo thật |
| Decode một request | Speculative trong pipeline (1.35–2.15×) | n-gram lookahead, mỗi đoán một micro-batch | **Có ý tưởng, chưa hiệu quả** |
| Nhiều request | Batching / continuous batching (exo 2.2×) | Mỗi request chạy riêng, khoá forward | **Thiếu** |
| Chịu lỗi giữa request | Petals: bản sao + phát lại input | Gửi full context (dựng lại KV được); một chủ cho mỗi đoạn; file trọng số chỉ nằm trên máy ghi | **Thiếu bản sao** |
| Máy thiếu RAM | prima.cpp mmap + prefetch; TPI-LLM sliding window | Không | Ngoài phạm vi hiện tại |
| Khám phá / định tuyến | Petals DHT; exo tự phát hiện | Bridge P2P, bầu master, handshake có version | Đủ cho LAN |
| Bảo mật | Petals: cảnh báo stage đầu thấy được token | TLS tuỳ chọn | Nên ghi rõ vấn đề riêng tư như Petals |
| Huấn luyện | Petals: fine-tune PEFT qua swarm | 1F1B, gradient liên stage thật (có giới hạn) | Hiếm hệ thống edge có; đáng nêu |
| Đánh giá | Máy thật + mạng giả lập, TTFT/TPOT, so với 1 máy và offload | Bảng speedup từ mô hình thời gian mô phỏng; parity đúng đắn; chạy thật nhiều process trên 1 máy | **Thiếu số đo thật** |

## 5. Việc nên làm, xếp theo lợi ích so với công sức

1. **Sửa cách trình bày trong README.**
   - Bỏ hoặc giới hạn câu "PIPELINE BUBBLE-FREE… 333 tokens/sec" và "True Distributed Supercomputing" cho trường hợp decode một request.
   - Nói rõ như exo và llama.cpp: PP cho dung lượng và thông lượng, không tăng tốc một request.
2. **Benchmark thật theo khung Petals/prima.cpp.**
   - Đo TTFT, TPOT, tok/s cho 1 request và cho N request đồng thời.
   - So với 1 máy (khi model vừa) và với offload (khi không vừa).
   - Chạy ở ba mức mạng: LAN 1 Gbit, 100 Mbit / 5 ms, 100 Mbit / 100 ms. Linux dùng `tc netem`; Windows dùng clumsy.
   - Dùng model đủ lớn (≥ 1–3B). Với 135M, thời gian mạng lấn át tất cả.
3. **Batching nhiều request qua pipeline.** Đây là nơi PP thực sự có lợi (exo 2.2× với 3 máy).
4. **Verify nhiều token đoán trong một forward, và huỷ micro-batch lỗi thời.** Hướng PipeInfer/FlowSpec.
5. **Partition bằng mô hình độ trễ đo thật + DP (EdgeShard) hoặc ILP (Halda).** Thay trọng số heuristic.
6. **Lượng tử trọng số 8/4-bit có kiểm chứng.** Mục đích là giảm số stage.
7. **Bản sao đoạn layer + định tuyến theo request (Petals/Parallax)** để chịu lỗi giữa chừng.

## Nguồn
- exo: https://github.com/exo-explore/exo · benchmark: https://blog.exolabs.net/day-1/ · issue chia layer: https://github.com/exo-explore/exo/issues/284
- Petals: https://arxiv.org/abs/2209.01188 · NeurIPS 2023: https://arxiv.org/abs/2312.08361
- distributed-llama: https://github.com/b4rtaz/distributed-llama
- llama.cpp RPC: https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md
- prima.cpp: https://arxiv.org/abs/2504.08791
- TPI-LLM: https://arxiv.org/abs/2410.00531
- Galaxy: https://arxiv.org/abs/2405.17245
- EdgeShard: https://arxiv.org/abs/2405.14371
- Helix: https://arxiv.org/abs/2406.01566
- HexGen: https://arxiv.org/abs/2311.11514
- Parallax: https://arxiv.org/abs/2509.26182
- PipeInfer: https://arxiv.org/abs/2407.11798 · FlowSpec: https://arxiv.org/abs/2507.02620 · SpecPipe: https://arxiv.org/abs/2504.04104
- Nén giao tiếp: https://arxiv.org/abs/2411.09510
