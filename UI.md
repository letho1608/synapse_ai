# UI.md — Đặc tả giao diện console synapse (học Claude Code + OpenCode, KHÔNG agent)

**Nguồn tham khảo:** `anomalyco/opencode` (TUI: slash menu, `ctrl+x` leader, palette, picker, status) và `anthropics/claude-code` (transcript + prompt ghim đáy + status line, tiền tố `/` `!` `@`, dialog, phím).
**Nguyên tắc:** chỉ mượn VỎ (bố cục, menu, phím, phản hồi), ruột vẫn là lệnh deterministic cũ — mọi dòng in ra đều là kết quả lệnh thật hoặc log thật, không chữ nào do AI bịa.
**Hiệu đính sau đối chiếu source/docs 2 repo (2026-09-12):** bản đầu thiếu `!` shell escape, history lưu file, banner khởi động, `/export` + `/bug`, palette `//`, và diễn đạt sai 1 chỗ (Enter khi nhiều kết quả PHẢI hiện list chọn số, không được tự chạy khớp dài nhất — cả 2 repo đều bắt chọn rõ ràng). Đã sửa hết dưới đây.

---

## 1. Bố cục màn hình (2 cột: trái hội thoại, phải vận hành)

```
┌─ chat + thẻ lệnh ─────────────┬─ cluster ─────────────┐
│ > mấy node đang online?       │ ● node-01  leader     │
│                               │ ● node-02  ok  41t/s  │
│   3 node, 1 leader…           │ ● node-03  ok  38t/s  │
│                               │ model SmolLM2-135M    │
│ ▸ /benchmark --reps 20        ├─ jobs ────────────────┤
│   tok/s 41.2 · failover ok    │ (trống khi rảnh)      │
│ ✔ exit 0 · 12.3s              ├─ logs ────────────────┤
│                               │ [node] heartbeat ok   │
├─ ô nhập (slash menu) ─────────┤ [node] shard 4/9      │
│ status bar────────────────────┴───────────────────────│
└───────────────────────────────────────────────────────┘
```

* **Trái = cái đã xong + hội thoại** (chat Q&A, thẻ kết quả lệnh, theo thời gian). **Phải = cái đang sống** (cluster map, jobs đang chạy, đuôi log) — tự refresh, không cần gõ.
* Ô nhập full-width, **chat-first**: chữ thường gửi thẳng cho model; `/lệnh`, `!shell`, `?`, `//` giữ nguyên.
* Terminal hẹp (<90 cột): panel phải tự ẩn, về 1 cột.
* Pipe/test/`--line`/`SYNAPSE_LINE=1` → line-mode 1 cột cũ (prefix form dưới đây).
* **Form dòng in (cả 2 mode, TUI tô màu theo theme):**

| Tiền tố | Nghĩa | Màu TUI |
|---|---|---|
| `> …` | Lệnh đã gõ (echo) | accent |
| `[node] …` | Log backend | mờ |
| `[console] …` | Vòng đời console (boot/stop/ngắt) | chữ thường |
| `[shell] …` + `[exit N]` | Shell escape | chữ thường |
| `[export]/[bug]/[logs] …` | Xác nhận ghi file/đổi trạng thái | chữ thường |
| `[ok]` / `PASS` | Thành công | xanh lá |
| `[fail]` / `FAIL` | Thất bại | đỏ |
| text thuần | Kết quả lệnh (parse được) | chữ thường |
* Banner khởi động (học cả 2 repo đều chào + mách nước khi mở): version, model hiệu dụng, cổng API/P2P, 3 lệnh gợi ý (`/status`, `/topology`, `/help`), 1 dòng nhắc `/quit` thoát sạch. Không banner rườm rà nhiều màn hình.

## 2. Ô nhập chat-first + menu `/` lọc dần

* **Chữ thường = chat:** gửi thẳng cho model (như `/client <text>` cũ), đáp án stream về transcript trái. Không còn "plain text does nothing".
* `/xyz` không tồn tại → báo không hiểu + không gọi hàm nào (không đoán).
* `!` shell, `?` phím tắt, `//` palette: giữ nguyên.
* Gõ `/` + chữ: hiện ô menu khớp (fuzzy, không phân biệt hoa thường).
  VD `/to` → `/topology`; `/mo` → `/models`, `/model` không tồn tại → gợi ý gần nhất.
* Tab = tự hoàn thành (dùng `readline` có sẵn, không dep mới).
* Enter khi còn nhiều kết quả: HIỆN DANH SÁCH ĐÁNH SỐ, KHÔNG chạy gì cả — user gõ số mới chạy (cả 2 repo đều bắt chọn rõ ràng, không bao giờ tự chạy khớp dài nhất).
* Gõ `/` trơ trụi + Enter: liệt kê toàn bộ lệnh + mô tả 1 dòng (thay `/help`). Bí danh `//` + Enter mở cùng danh sách này (tương đương command palette `ctrl+p` của OpenCode — `ctrl+p` trong readline đã là lịch sử lệnh nên không cướp).
* **Chế độ TUI:** gõ `/` là ô menu xổ ngay trên ô nhập — kể cả `/` trơ trụi cũng hiện đủ lệnh (học autocomplete của OpenCode: box riêng, lọc fuzzy từng ký tự, có mô tả); `Tab`/`Enter` chèn `/lệnh + space`, `Esc` ẩn, gõ tiếp args thì box tự ẩn. `Ctrl+P` hoặc `C-x p` hoặc `//` mở palette (gõ lọc, lên/xuống, Enter chọn); mơ hồ thì dialog chọn thay list số.
* Chữ không slash, không `!`: báo "không hiểu, gõ /help", KHÔNG làm gì.
* Lịch sử lệnh LƯU FILE (`~/.synapse/console_history`, readline read/write) — mở lại console vẫn còn mũi tên lên/xuống (tương đương gần nhất với "sessions" của họ mà không cần agent session).
* Tiền tố `!` (học Claude shell-mode, rất hợp nghề vận hành): `!nvidia-smi`, `!netstat -ano | findstr 50051` — chạy lệnh shell local, in output inline ngay dưới, không gửi đi đâu, không qua agent. Timeout 30s, exit code in kèm. Cấm mặc định các lệnh phá hoại? Không — user tự chịu như shell thật, nhưng `rm -rf /`, `mkfs`, `dd` bị chặn cứng theo allowlist-phủ-định tối thiểu (ghi rõ trong help).

Danh sách lệnh (giữ nguyên ý nghĩa cũ, chỉ thêm 3 lệnh mục 5):

| Lệnh | Việc | Ghi chú |
|---|---|---|
| `/status [--json] [--port N]` | Sức khỏe node + cụm | Bảng rich (mục 4) |
| `/topology [--json]` | Bản đồ chia việc | Bảng rich |
| `/models [list\|pull\|rm\|use] [id]` | Kho model | Bảng rich; pull/rm/unset/use hỏi `y/N` trước; `use` = set default + hẹn reboot (KHÔNG đổi nóng — đổi model phải reboot node, nói rõ chứ không giả vờ như picker của họ) |
| `/benchmark [--max-tokens N] [--reps N] [--failover]` | Đo cụm | Progress % (mục 4) |
| `/client <text...>` | Chat 1 câu | Stream chữ ra dần |
| `/export [file]` | Xuất transcript (lệnh đã gõ + kết quả + log) ra markdown | MỚI — học `/export` của OpenCode; mặc định `synapse-export-<giờ>.md` ở CWD |
| `/bug` | Gom bundle chẩn đoán: status json + versions + 200 dòng log cuối + cổng | MỚI — học `/bug` của Claude; in đường dẫn file, không tự gửi đi đâu |
| `/set <key> <value>` | Lưu default | Validate như hiện tại |
| `/config` | Bảng giá trị + nguồn | Bảng rich |
| `/unset <key>` | Xóa default | Hỏi `y/N` nếu key đang có giá trị |
| `/logs [on\|off]` | Tắt/bật stream log `[node]` | MỚI (mục 5) |
| `/clear` | Xóa màn hình | MỚI (mục 5) |
| `/doctor` | Khám máy: cổng, version, TLS, model cache | MỚI (mục 5) |
| `/theme [name]` | Đổi theme màu (`synapse`, `ember-dark`, `ember-light`, custom) | MỚI (mục 10); TUI đổi ngay, line-mode boot sau |
| `/help` | Bảng lệnh + phím | Bảng rich |
| `/quit`, `/exit` | Dừng node, thoát 0 | Giữ nguyên |

## 3. Thẻ lệnh + panel phải (TUI)

* Mỗi lệnh chạy là 1 thẻ ở cột trái: header `▸ /lệnh … · bắt đầu HH:MM:SS`, output thu gọn (>30 dòng tự gập, Enter mở), footer `✔ exit N · X.Xs` / `✘ ngắt bởi user` / `[fail] …`.
* Lệnh đang chạy: panel phải mục jobs hiện `▸ /lệnh … ⏳ Xs`, transcript trái chỉ 1 dòng header. Xong → footer về trái, jobs hết.
* Panel phải (rộng 40, ẩn khi terminal <90 cột): cluster (từng peer: role, tok/s — best effort, thiếu thì `?`), model hiệu dụng, jobs, 8 dòng log cuối (mờ). Refresh mỗi giây khi có jobs, 5 giây khi rảnh.
* `/logs on|off`: ẩn/hiện vùng logs của panel (tắt vẫn buffer 200 dòng, bật lại xả).
* `/benchmark` dùng model hiệu dụng (không bao giờ `default` lạ backend 400); trong TUI output harness được capture vào thẻ (tiến trình con không được vẽ đè lên app).
* Line-mode: không panel; thẻ lệnh = header/footer text thuần cùng prefix form (§1).

## 4. Thanh trạng thái + line-mode fallback

* TUI: status bar 1 dòng ghim đáy (node | peers | model | modes) + `?` gợi ý phím. Refresh 5s; API chết → `peers ?`.
* Line-mode (pipe/`TERM=dumb`): status line in trước mỗi prompt (cơ chế ANSI cũ). Mọi quy tắc menu/chat/thẻ lệnh giữ nguyên hành vi, chỉ khác vẽ.

## 4. Bảng màu + tiến độ (dùng `rich`, đã là dependency)

* `/status`, `/topology`, `/models`, `/config`, `/help`: render `rich.table.Table` thay vì text thô. Fallback text thuần khi stdout không phải TTY (pipe sang file vẫn parse được).
* `/benchmark`: `rich.progress` (%, tok/s hiện tại, ETA); xong in bảng tổng kết. `--json` giữ nguyên 1-document JSON (không vẽ bảng).
* `/client`: stream token ra dần (đã có), thêm spinner khi chờ token đầu (TTFT).
* `/models pull|rm`, `/unset` có giá trị: confirm `y/N` mặc định N (học permission prompt của Claude, nhưng là hỏi cứng 1 câu, không mode phức tạp).

## 5. Ba lệnh mới đúng nghề vận hành

* **`/logs [on|off]`** — không args = đảo trạng thái. Tắt: log `[node]` dồn vào bộ đệm vòng 200 dòng (không mất); bật lại: xả bộ đệm rồi stream tiếp. In trạng thái hiện tại sau mỗi lần đổi.
* **`/clear`** — xóa màn hình (`os.system clear/cls`, fallback in 50 dòng trống), giữ nguyên log backend + lịch sử lệnh + defaults. Không đụng server.
* **`/doctor`** — chạy kiểm tra và in bảng PASS/FAIL từng mục, exit không đổi (chỉ khám, không sửa):
  1. Cổng p2p/API có bind được không (`check_ports_free` có sẵn).
  2. API local có trả lời `/healthcheck` không.
  3. Version handshake backend (`VERSION`) vs console.
  4. TLS mode hiện tại (`SYNAPSE_TLS`) + cert files có tồn tại không.
  5. Model mặc định có trong cache không (đỡ bị bất ngờ tải ở request đầu).
  6. Discovery module + Tailscale CLI có mặt không.
  Mỗi FAIL kèm 1 dòng cách sửa (copy được lệnh gợi ý).

## 6. Bàn phím (khớp cả 2 repo, trung thực từng phím)

| Phím | Việc | Trạng thái |
|---|---|---|
| Enter | Gửi lệnh | Có |
| Lên/Xuống | Lịch sử lệnh (readline, lưu file) | Có |
| Ctrl+R | Tìm ngược lịch sử (readline) | Có sẵn, phải ghi vào `/help` |
| Ctrl+C (in command) | Ngắt THẬT: kill benchmark/shell con, đóng HTTP đang treo, giữ phiên | Có (cả 2 mode; pull thì server có thể pull nốt phía sau) |
| Ctrl+C lần 2 / `/quit` | Dừng node, thoát 0 | Có |
| Ctrl+L | Vẽ lại + hiện gợi ý `/clear` (học Claude) | MỚI, dễ |
| Tab | Hoàn thành lệnh slash | MỚI (readline completer) |
| `?` khi prompt trống | Bảng phím tắt (học cả 2 repo đều có `?`) | MỚI, dễ |
| Esc | Ngắt như Ctrl+C (readline thuần không bắt được Esc đơn — document rõ) | Hạn chế đã biết (line-mode); TUI: Esc đóng overlay |
| Alt+Enter / Ctrl+J | Xuống dòng trong ô nhập (TUI) | MỚI — Enter gửi, Alt+Enter xuống dòng |
| Ctrl+P | Palette lệnh (TUI) | MỚI — học `command_list` của OpenCode |
| Ctrl+X P / M / T / ? | Palette / picker model / theme / help (TUI leader) | MỚI — học `keybinds.leader: ctrl+x` của OpenCode |
| Ctrl+X Ctrl+C | Thoát (TUI leader) | MỚI |
| Ctrl+D (ô nhập TUI) | Thoát | MỚI |
| `ctrl+x ...` leader, `ctrl+p` palette, `ctrl+o` model dialog, `ctrl+s` sessions | Cả chùm phím gảy App của OpenCode | KHÔNG phase 1 — readline không làm được chuỗi 2 phím; chờ `prompt_toolkit` (phase 2). Không giả vờ có. |

## 7. Phase 2 — full-screen (prompt_toolkit, ĐÃ LÀM 2026-09-12)

* `backend/synapse/tui.py`: custom Application — transcript cuộn + ô nhập nhiều dòng (Enter gửi, Alt+Enter/Ctrl+J xuống dòng) + toolbar 2 dòng (status + gợi ý leader) + float overlay cho palette/picker/theme/help/confirm. `run_console` tự vào TUI khi stdout là TTY; pipe/test/`--line`/`SYNAPSE_LINE=1`/thiếu dep → line-mode.
* TUI mở trước, backend boot nền sau: mở lên là theme ngay (`[console] starting node…` + `node running (pid …)` hiện trong transcript, không còn dòng text trước theme).
* Leader `Ctrl+X` đúng OpenCode (`c-x c-c` thoát, `c-x p` palette, `c-x m` picker model, `c-x t` theme, `c-x ?` help); `Ctrl+P` palette; picker lọc 510 model catalog (gõ lọc, lên/xuống, Enter = `/models use` + hẹn reboot); confirm pull/rm/use/unset bằng dialog y/n (thay hỏi text).
* Giới hạn còn lại (ghi rõ): `C-x` đơn không có tác dụng (chỉ chuỗi); editor xuống dòng nối bằng space khi gửi (hợp lệnh nhiều dòng `/client` dài); ngắt `/models pull` chỉ dừng phía client — server có thể pull nốt phía sau.
* Tuyệt đối KHÔNG mang về kể cả phase 2: sessions/resume/undo/share/themes, `@` file-mention (lệnh mình không nhận arg node/file nên mention vô nghĩa), đổi model nóng (picker chỉ được set default + hẹn reboot).

## 8. Không làm (ranh giới vỏ/agent)

* Không đoán ý định, không gọi model để diễn giải lệnh, không tự chuỗi nhiều lệnh.
* Không session hội thoại, không lịch sử ngữ nghĩa — lịch sử chỉ là dòng lệnh đã gõ.
* Không plugin/store/skill/MCP.
* Mọi hành động đổi cụm (pull/rm/unset/benchmark --failover) đều confirm `y/N` mặc định N.

## 9. Test chấp nhận (mỗi mục 1 test behavioral, không soi source)

1. `/` + Enter liệt kê đủ lệnh trong bảng mục 2.
2. `/to` + Tab ra `/topology`; `/xyz` báo không hiểu + không gọi hàm nào.
3. Status line: API chết → thanh đáy in `peers ?`, prompt vẫn hiện trong 2s; log tới giữa lúc gõ → buffer đang gõ còn nguyên.
4. Pipe stdout (không TTY): bảng rich tắt, text thuần parse được.
5. `/models pull x` dừng ở `y/N`, gõ Enter (=N) thì không gọi API.
6. `/logs off` → log dồn bộ đệm; `/logs on` → xả đủ, không mất dòng.
7. `/doctor` trên máy lỗi cổng: FAIL đúng mục + dòng sửa chạy được.
8. `/clear` không giết backend, không mất history/defaults.
9. Ctrl+C giữa benchmark: dừng gọn, exit ở lại console.
10. `!echo hi` in `hi` + exit code; `!rm -rf /` bị chặn cứng + báo rõ; lệnh treo quá 30s bị kill.
11. Khởi động lại console: mũi tên lên còn lệnh cũ (history file).
12. `/export` ra file markdown mở được; `/bug` ra file bundle có đủ status + 200 log cuối.
13. Phase 2 (khi làm): menu live lọc đúng, Esc đóng, Enter chạy đúng lệnh đã chọn.
14. `/theme nope` báo fail + liệt kê theme có; `/theme ember-dark` lưu xong `current` đúng.
15. Theme custom file sai JSON / token lạ: load vẫn ra theme hợp lệ (fallback builtin, không crash).
16. Ctrl+C giữa `!sleep`: tiến trình con chết thật, rc 130, phiên còn, không orphan.
17. Ctrl+C giữa stream treo/API treo: `Cancelled`, về prompt trong vài giây, worker không crash loop.
18. Chữ thường (không `/`) → đi chat (gọi client 1 lần); `/xyz` → không gọi gì.
19. Thẻ lệnh: output có header `▸ …` + footer `exit N · Xs`; panel render đủ 3 mục cluster/jobs/logs.

## 10. Theme màu (học `/theme` của cả 2 repo)

* Builtin: `synapse` (mặc định, dark kiểu Tokyonight), `ember-dark` (accent cam `#d77757`, success `#4eba65`, error `#ff6b80` — tông Claude), `ember-light` (success `#2c7a39`, error `#ab2b3f`).
* Custom: file `~/.synapse/themes/<ten>.json` dạng `{name, base, overrides}` (schema Claude) + `defs`/tham chiếu/`{"dark","light"}`/`none`/ANSI (schema OpenCode); token lạ hoặc JSON hỏng bị bỏ qua, không bao giờ crash UI.
* Chọn: `/theme` (liệt kê) hoặc `/theme <ten>` hoặc overlay `C-x t`. Lưu ở `~/.synapse/theme.json`; TUI đổi màu ngay, line-mode boot sau.
