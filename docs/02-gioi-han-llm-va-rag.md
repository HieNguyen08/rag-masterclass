# Module 02 — Giới hạn LLM & hình thức hóa RAG

> Thời lượng: ~40 phút · Mức độ: Trung bình · Tiên quyết: Module 01 (tokenization, attention, cross-entropy, decoding, post-training)

Module 01 cho ta thấy LLM là một hàm $p_\theta(t_i \mid t_{<i})$ được huấn luyện để *giống văn bản trong dữ liệu*. Module này hỏi ngược lại: **chính vì được xây như vậy, LLM sẽ hỏng ở đâu khi ta đặt nó vào vị trí trả lời email khách hàng thay cho nhân viên CS?** Sau khi liệt kê các giới hạn (hallucination, knowledge cutoff, thiếu tri thức riêng, context dài kém hiệu quả và đắt), ta so sánh các cách khắc phục (prompt, RAG, fine-tune, long-context, CAG), rồi **hình thức hóa RAG** bằng xác suất — từ RAG gốc của Lewis et al. (2020), REALM, Fusion-in-Decoder đến RETRO, Atlas — và kết thúc bằng bản đồ các điểm hỏng của một pipeline RAG thực tế.

Module này là "khung lý thuyết" cho phần còn lại của khóa. Mỗi điểm hỏng ở cuối module được giải quyết trong một module cụ thể; mình sẽ ghi rõ dẫn chiếu.

## Mục tiêu học tập

Sau module này, bạn có thể:

1. Phân loại một câu trả lời sai của LLM thành intrinsic / extrinsic, factuality / faithfulness, và chỉ ra nguyên nhân khả dĩ (dữ liệu, mục tiêu huấn luyện, decoding, cách đánh giá).
2. Ước lượng (bằng số) chi phí và rủi ro chất lượng của cách "nhồi toàn bộ kho tri thức vào context" so với RAG cho tải 1.500 ticket/ngày.
3. Dùng ma trận quyết định để chọn giữa prompt-only, RAG, fine-tune, long-context, CAG cho từng loại tri thức của bài toán Zendesk, và biện luận bằng bằng chứng nghiên cứu.
4. Viết và giải thích công thức RAG-Sequence, RAG-Token; tính tay một ví dụ; giải thích vì sao REALM coi tài liệu truy xuất là biến ẩn và gradient được trọng số theo posterior.
5. Ánh xạ bảy điểm hỏng của Barnett et al. (2024) vào pipeline Zendesk và chỉ ra module nào xử lý từng điểm.
6. Lập luận vì sao bài toán Zendesk là **RAG + phân loại + quyết định escalate**, không phải "chỉ RAG".

---

## 1. Hallucination — khi model nói trôi chảy nhưng sai

### 1.1 Định nghĩa và phân loại

Trong NLG, **hallucination** là nội dung được sinh ra trôi chảy, có vẻ hợp lý, nhưng *không trung thành với nguồn* hoặc *không đúng sự thật*. Có hai trục phân loại hay dùng, bạn nên nắm cả hai vì tài liệu khác nhau dùng khác nhau.

**Trục 1 — so với nguồn đầu vào** (Maynez et al., 2020, trong tóm tắt văn bản; Ji et al., 2023, mở rộng cho NLG nói chung):

- **Intrinsic hallucination:** đầu ra *mâu thuẫn* với nguồn được cung cấp. Ví dụ: tài liệu chính sách nói "hoàn tiền trong 14 ngày", model viết "hoàn tiền trong 30 ngày".
- **Extrinsic hallucination:** đầu ra chứa thông tin *không kiểm chứng được* từ nguồn — không mâu thuẫn nhưng cũng không có căn cứ. Ví dụ: tài liệu không nhắc tới phí, model thêm "không mất phí xử lý". Thông tin extrinsic có thể tình cờ đúng, nhưng với hệ thống CS thì "tình cờ đúng" vẫn là lỗi quy trình.

<!-- fig:intrinsic-extrinsic -->
<figure markdown="span">
  ![Trục 1 phân loại hallucination theo quan hệ với nguồn được cung cấp, minh họa bằng ví dụ chính sách hoàn tiền của mục 1](assets/figures/02/intrinsic-extrinsic.light.svg#only-light){ loading=lazy }
  ![Trục 1 phân loại hallucination theo quan hệ với nguồn được cung cấp, minh họa bằng ví dụ chính sách hoàn tiền của mục 1](assets/figures/02/intrinsic-extrinsic.dark.svg#only-dark){ loading=lazy }
  <figcaption>Hình 2.1 — Trục 1 phân loại hallucination theo quan hệ với nguồn được cung cấp, minh họa bằng ví dụ chính sách hoàn tiền của mục 1.1.</figcaption>
</figure>
<!-- /fig -->

**Trục 2 — dành cho LLM** (Huang et al., 2023, survey):

- **Factuality hallucination:** sai so với sự thật thế giới (bịa sự kiện, bịa trích dẫn, sai ngày).
- **Faithfulness hallucination:** không trung thành với *chỉ dẫn* của người dùng (làm khác yêu cầu), với *context* được cung cấp, hoặc *tự mâu thuẫn* (logic không nhất quán trong chính câu trả lời).

Với RAG, **faithfulness so với context** là thứ ta kiểm soát được và đo được (Module 07, 10). Factuality phụ thuộc vào việc *tài liệu* có đúng và mới không — bài toán dữ liệu (Module 04).

| Ví dụ trong email AI soạn | Trục 1 | Trục 2 |
|---|---|---|
| Tài liệu nói gói Pro có 10 user; AI viết 20 | Intrinsic | Faithfulness (context) |
| Không có tài liệu về SLA gói Enterprise; AI hứa "phản hồi trong 1 giờ" | Extrinsic | Factuality (và vi phạm chính sách không bịa SLA) |
| Khách hỏi bằng tiếng Nhật, AI trả lời tiếng Anh dù system prompt yêu cầu trả lời cùng ngôn ngữ | — | Faithfulness (chỉ dẫn) |
| Đoạn đầu email nói "đã hoàn tiền", đoạn sau nói "sẽ xem xét hoàn tiền" | — | Faithfulness (tự mâu thuẫn) |

### 1.2 Vì sao LLM hallucinate — bốn nhóm nguyên nhân

**(a) Dữ liệu.** Tri thức hiếm được học kém. Kandpal et al. (2023) cho thấy độ chính xác trả lời câu hỏi sự kiện của LLM tương quan mạnh với *số tài liệu liên quan* tới câu hỏi trong dữ liệu pretraining: sự kiện xuất hiện ít ("đuôi dài") được trả lời kém hơn hẳn, và tăng kích thước model chỉ cải thiện chậm ở vùng này; retrieval giúp đáng kể. Tri thức nội bộ của doanh nghiệp bạn là trường hợp cực đoan của đuôi dài: tần suất xuất hiện trong dữ liệu pretraining bằng 0.

**(b) Mục tiêu huấn luyện.** Cross-entropy (Module 01, mục 5) thưởng cho việc gán xác suất cao cho văn bản *giống dữ liệu*, không có khái niệm "đúng" hay "tôi không biết". Khi gặp câu hỏi về điều chưa từng thấy, phân phối "giống văn bản" nhất thường là một câu trả lời tự tin có hình thức hợp lý.

**(c) Đánh giá và post-training thưởng cho việc đoán.** Kalai et al. (2025) lập luận rằng hallucination có nguồn gốc thống kê khá "bình thường": lỗi sinh văn bản có thể được chặn dưới bởi tỉ lệ lỗi của một bài toán phân loại nhị phân tương ứng ("câu này có hợp lệ không?"); và với các sự kiện tùy ý chỉ xuất hiện rất ít lần trong dữ liệu, model khó tránh được lỗi. Điểm thứ hai của họ quan trọng hơn cho người làm sản phẩm: **phần lớn benchmark chấm đúng/sai nhị phân, không cho điểm câu "tôi không biết"**, nên quá trình tối ưu theo benchmark khuyến khích model đoán.

Ta có thể tự dẫn ra hiện tượng đó bằng một mô hình đơn giản. Giả sử với một câu hỏi, model tin rằng câu trả lời của nó đúng với xác suất $p$. Hệ chấm điểm cho $+1$ nếu đúng, $-c$ nếu sai ($c \ge 0$), $0$ nếu từ chối. Kỳ vọng điểm khi trả lời là

$$
\mathbb{E}[\text{điểm} \mid \text{trả lời}] = p - c(1-p),
$$

còn khi từ chối là $0$. Trả lời có lợi khi và chỉ khi

$$
p - c(1-p) > 0 \iff p > \frac{c}{1+c}.
$$

- Benchmark nhị phân thông thường: $c = 0$ → ngưỡng $0$ → **luôn nên đoán**, dù $p = 0.05$.
- Nếu phạt sai bằng điểm thưởng đúng ($c = 1$) → chỉ trả lời khi $p > 0.5$.
- Nếu phạt sai gấp 9 lần ($c = 9$) → chỉ trả lời khi $p > 0.9$.

<!-- fig:abstain-threshold -->
<figure markdown="span">
  ![Mô hình chấm điểm của mục 1](assets/figures/02/abstain-threshold.light.svg#only-light){ loading=lazy }
  ![Mô hình chấm điểm của mục 1](assets/figures/02/abstain-threshold.dark.svg#only-dark){ loading=lazy }
  <figcaption>Hình 2.2 — Mô hình chấm điểm của mục 1.2: trả lời có lợi khi đường nằm trên 0, tức p > c/(1 + c). Với c = 0 thì luôn nên đoán.</figcaption>
</figure>
<!-- /fig -->

Đây là toán của **selective prediction**, và nó trùng khớp với quyết định escalate trong Zendesk: một email trả lời sai về hoàn tiền tốn kém hơn nhiều so với lợi ích một email đúng, nên ngưỡng tự gửi phải rất cao. Module 10 sẽ biến $c$ thành ma trận chi phí thật và $p$ thành xác suất đã hiệu chuẩn.

**(d) Decoding và ngữ cảnh.** Lấy mẫu với temperature cao tăng khả năng chọn token xác suất thấp; một khi một token sai đã được sinh, các token sau *điều kiện trên* nó (chain rule) và có xu hướng "bảo vệ" lỗi đó cho nhất quán (snowballing). Ngữ cảnh dài, nhiễu, có tài liệu mâu thuẫn cũng làm model dễ lấy nhầm thông tin (mục 3).

**Fine-tune trên tri thức mới có thể làm tệ hơn.** Gekhman et al. (2024) thiết kế thí nghiệm có kiểm soát: fine-tune model trên tập ví dụ trong đó một phần chứa sự kiện model *chưa biết*. Kết quả: model học các ví dụ "mới" chậm hơn nhiều so với ví dụ phù hợp tri thức sẵn có, và khi cuối cùng đã khớp được chúng, xu hướng hallucinate (so với tri thức trước đó) tăng lên. Thông điệp: fine-tune giỏi dạy *cách dùng* tri thức có sẵn, kém trong việc *nạp* tri thức mới.

**Hallucination không thể loại bỏ hoàn toàn.** Một số tác giả (ví dụ Xu et al., 2024) đưa ra lập luận hình thức rằng với mọi LLM tính được, luôn tồn tại đầu vào khiến nó sai. Dù mức độ áp dụng thực tế của các kết quả như vậy còn được tranh luận, hệ quả kỹ thuật không đổi: **thiết kế hệ thống phải giả định model sẽ sai một tỉ lệ nào đó**, và đặt các lớp phát hiện + chuyển người xung quanh nó.

### 1.3 Liên hệ Zendesk — hallucination nào nguy hiểm nhất?

Không phải mọi hallucination có chi phí như nhau. Với case study (giả định: SaaS B2B, khách Việt/Nhật/quốc tế):

| Loại nội dung bị bịa | Hậu quả | Mức nghiêm trọng |
|---|---|---|
| Giá, khuyến mãi, hoàn tiền, cam kết SLA | Ràng buộc pháp lý/tài chính, khách khiếu nại | Rất cao → luôn grounding + guardrail + thường escalate |
| Thông tin tài khoản cụ thể ("đơn của anh đã được xử lý") | Sai sự thật về khách cụ thể; có thể lộ dữ liệu tenant khác | Rất cao → chỉ lấy từ API/tool, không từ LLM |
| Bước hướng dẫn sản phẩm (menu, cài đặt) | Khách làm theo không được, reopen ticket | Trung bình → RAG từ Help Center, kiểm citation |
| Văn phong, lời chào, câu đệm | Thường vô hại | Thấp |

Mình khuyên đặt mục tiêu theo *loại* chứ không theo tổng: "0 câu bịa chính sách trong 1.000 email shadow-mode" có ý nghĩa vận hành hơn "tỉ lệ hallucination 2%".

---

## 2. Knowledge cutoff và tri thức riêng tư

### 2.1 Knowledge cutoff

Tham số của LLM được "đóng băng" tại thời điểm dữ liệu huấn luyện được thu thập. Mọi sự kiện sau đó — phiên bản API mới, tính năng vừa ra, giá vừa đổi — model không biết. Tệ hơn, model thường *không biết rằng mình không biết*: hỏi về "phiên bản 5.2" của sản phẩm, model có thể trả lời bằng kiến thức về 4.x với giọng tự tin.

Hai chi tiết thực tế:

- Ngày cutoff công bố thường là ngày *thu thập* dữ liệu; mật độ dữ liệu ở những tháng sát cutoff thường thưa (internet chưa kịp viết về sự kiện), nên tri thức "gần cutoff" kém hơn tri thức cũ.
- Model thương mại được cập nhật theo lịch của nhà cung cấp, không theo lịch của bạn. Kho tri thức Zendesk (giả định) thay đổi **hằng tuần** — không có cách nào để tham số model theo kịp.

### 2.2 Tri thức riêng tư / doanh nghiệp

Phần lớn tri thức cần để trả lời ticket **chưa bao giờ** nằm trên internet công khai:

- ~800 bài Help Center (có thể công khai, nhưng phiên bản mới nhất và bài nội bộ thì không),
- ~300 macro của đội CS,
- ~200.000 ticket đã giải quyết (chứa PII, chứa cách xử lý ngoại lệ),
- chính sách giá/hoàn tiền/SLA hiện hành, release notes, tài liệu API nội bộ,
- và *dữ liệu của chính khách hàng đang hỏi* (gói dịch vụ, hóa đơn, trạng thái tài khoản) — chỉ lấy được qua API tại thời điểm trả lời.

Ngoài chuyện "không biết", còn chuyện **không được biết**: tri thức của tenant A không được xuất hiện trong câu trả lời cho tenant B. Nếu nạp tri thức vào tham số (fine-tune), ta mất khả năng kiểm soát truy cập theo từng request — một lý do kiến trúc, không chỉ là lý do chất lượng, để giữ tri thức *bên ngoài* model và truy xuất có kiểm soát quyền (ACL-aware retrieval, Module 05 và 11).

> **Liên hệ Zendesk.** Ba loại tri thức cần tách bạch ngay từ thiết kế: (1) **tri thức chung về sản phẩm** (Help Center, docs) — RAG; (2) **chính sách có hệ quả pháp lý** (giá, hoàn tiền, SLA) — RAG với nguồn "có thẩm quyền" duy nhất, phiên bản rõ ràng, và guardrail; (3) **dữ liệu theo khách hàng** (tài khoản, đơn, ticket trước) — tool/API call có kiểm tra quyền (Module 08), không index lẫn vào kho chung.

---

## 3. Context window: "đọc được" khác "dùng tốt"

Một cách "đơn giản" để vượt qua cutoff và tri thức riêng: đưa toàn bộ tri thức vào prompt. Tính đến 2026, nhiều model thương mại hàng đầu quảng cáo cửa sổ ngữ cảnh khoảng 1 triệu token, một số model mở còn lớn hơn. Câu hỏi là: model có *dùng tốt* thông tin ở mọi vị trí, và giá bao nhiêu?

### 3.1 Lost in the middle

Liu et al. (2024, TACL; preprint 2023) đặt một tài liệu chứa đáp án giữa nhiều tài liệu gây nhiễu (multi-document QA) và thay đổi *vị trí* của nó. Kết quả có hình chữ **U**: độ chính xác cao nhất khi tài liệu liên quan ở **đầu** hoặc **cuối** context, giảm rõ khi nó ở **giữa** — kể cả với các model được quảng cáo là long-context. Ở một số cấu hình, đặt đáp án ở giữa còn cho kết quả kém hơn cả việc không đưa tài liệu nào (closed-book).

<!-- fig:lost-in-middle -->
<figure markdown="span">
  ![Sơ đồ định tính của hiệu ứng lost in the middle (Liu et al](assets/figures/02/lost-in-middle.light.svg#only-light){ loading=lazy }
  ![Sơ đồ định tính của hiệu ứng lost in the middle (Liu et al](assets/figures/02/lost-in-middle.dark.svg#only-dark){ loading=lazy }
  <figcaption>Hình 2.3 — Sơ đồ định tính của hiệu ứng lost in the middle (Liu et al., 2024): đường cong chỉ thể hiện hình dạng, không phải số liệu trong bài báo.</figcaption>
</figure>
<!-- /fig -->

Một cách diễn giải trực giác dựa trên Module 01: causal attention + dữ liệu huấn luyện (thông tin quan trọng hay nằm đầu văn bản; token gần vị trí sinh thì có tín hiệu vị trí "gần") tạo thiên lệch vị trí; RoPE có xu hướng suy giảm theo khoảng cách. Đây là *giả thuyết giải thích*, không phải định lý — điều quan trọng với kỹ sư là **hiệu ứng được đo lặp lại nhiều lần**.

### 3.2 Độ dài hiệu dụng nhỏ hơn độ dài quảng cáo

- **RULER** (Hsieh et al., 2024) mở rộng bài test "kim trong đống rơm" (needle-in-a-haystack) thành nhiều loại tác vụ: nhiều kim, truy vết biến qua nhiều bước, tổng hợp, QA. Phát hiện chính: hầu hết model đạt gần tuyệt đối ở bài tìm kim đơn giản, nhưng hiệu năng tụt mạnh khi độ dài và độ khó tăng; độ dài "hiệu dụng" (giữ được chất lượng gần mức ở context ngắn) của nhiều model nhỏ hơn đáng kể so với con số công bố.
- **Context Rot** (Hong, Troynikov & Huber — báo cáo kỹ thuật của Chroma, 07/2025) thử 18 LLM và thấy chất lượng giảm khi độ dài đầu vào tăng, ngay cả với tác vụ đơn giản; giảm mạnh hơn khi câu hỏi và đáp án ít trùng từ ngữ (cần suy luận ngữ nghĩa), và khi có các đoạn gây nhiễu gần giống đáp án. Họ cũng ghi nhận các họ model khác nhau hỏng theo kiểu khác nhau: có họ thiên về từ chối, có họ thiên về trả lời sai một cách tự tin.

Lưu ý "gây nhiễu gần giống đáp án" chính là mô tả kho tri thức CS: 800 bài Help Center có hàng chục bài về đăng nhập, hàng chục bài về hóa đơn, nhiều phiên bản chính sách hoàn tiền theo thời gian.

### 3.3 Chi phí của context dài — tính bằng số

**Chi phí tính toán.** Với model $N$ tham số, $L$ lớp, chiều ẩn $d$, prefill một prompt $n$ token tốn xấp xỉ

$$
\text{FLOPs}_{\text{prefill}}(n) \approx \underbrace{2Nn}_{\text{các phép nhân ma trận trọng số}} + \underbrace{2Ln^2 d}_{\text{tính } QK^\top \text{ và nhân với } V},
$$

(số hạng attention viết gọn cho causal mask: $QK^\top$ và $AV$ mỗi cái khoảng $n^2 d$ phép nhân–cộng mỗi lớp nếu tính đủ ma trận, causal mask giảm một nửa; hằng số chính xác phụ thuộc cách cài đặt). Số hạng thứ nhất tuyến tính, số hạng thứ hai bậc hai. Ví dụ với cấu hình kiểu 7B ($N = 6.7\cdot10^9$, $L = 32$, $d = 4096$):

- $n = 8{,}000$: $2Nn \approx 1.07\cdot10^{14}$; $2Ln^2d \approx 1.68\cdot10^{13}$ → attention chiếm ~14%.
- $n = 640{,}000$: $2Nn \approx 8.6\cdot10^{15}$; $2Ln^2d \approx 1.07\cdot10^{17}$ → attention chiếm ~93%, tổng gấp ~930 lần so với 8K token.

Độ dài tăng 80 lần nhưng chi phí prefill tăng gần 1.000 lần. Thêm vào đó là bộ nhớ KV cache tăng tuyến tính theo $n$ (Module 11).

<!-- fig:prefill-flops -->
<figure markdown="span">
  ![Ước lượng FLOPs prefill của mục 3](assets/figures/02/prefill-flops.light.svg#only-light){ loading=lazy }
  ![Ước lượng FLOPs prefill của mục 3](assets/figures/02/prefill-flops.dark.svg#only-dark){ loading=lazy }
  <figcaption>Hình 2.4 — Ước lượng FLOPs prefill của mục 3.3 cho cấu hình kiểu 7B: phần attention bậc hai chiếm ~14% ở 8K token và ~93% ở 640K token.</figcaption>
</figure>
<!-- /fig -->

**Chi phí tiền.** Giả định (để học, không phải bảng giá thật):

- 1.500 ticket/ngày × ~3,5 lượt ≈ **5.250 lời gọi LLM soạn trả lời/ngày** (chưa tính các lời gọi phân loại, kiểm tra).
- Giá input giả định **\$1 / 1 triệu token** (giá thật thay đổi theo model; hãy thay số của bạn).
- Kho Help Center: 800 bài × ~800 token/bài ≈ **640.000 token** (ước lượng; chưa tính macro, chính sách, và tất nhiên không thể chứa 200.000 ticket lịch sử — khoảng $10^8$ token).

| Chiến lược | Token input/lời gọi | Token/ngày | Chi phí input/ngày (giả định) |
|---|---|---|---|
| RAG (system prompt + 5–8 chunk + thread + email) | ~6.000 | ~31,5 triệu | ~\$31,5 |
| Long-context: nhồi toàn bộ Help Center | ~646.000 | ~3,39 tỷ | ~\$3.390 |
| Long-context + prompt caching (đọc cache ở mức 0,1× giá input) | ~646.000 | ~3,39 tỷ | ~\$340 + chi phí ghi cache |

Prompt caching (ví dụ API của Anthropic tính đến 10/2026: đọc cache ~0,1× giá input với đa số model, ghi cache 1,25× cho TTL 5 phút hoặc 2× cho TTL 1 giờ) giảm chi phí đáng kể, nhưng (1) vẫn đắt hơn RAG khoảng 10 lần trong giả định này, (2) không giải quyết chất lượng (lost in the middle, context rot), (3) không giải quyết phân quyền (mọi tenant thấy cùng một kho), và (4) không chứa nổi ticket lịch sử.

<!-- fig:context-cost -->
<figure markdown="span">
  ![Chi phí input mỗi ngày theo bảng của mục 3](assets/figures/02/context-cost.light.svg#only-light){ loading=lazy }
  ![Chi phí input mỗi ngày theo bảng của mục 3](assets/figures/02/context-cost.dark.svg#only-dark){ loading=lazy }
  <figcaption>Hình 2.5 — Chi phí input mỗi ngày theo bảng của mục 3.3 (giá và lưu lượng đều là giả định để học).</figcaption>
</figure>
<!-- /fig -->

> **Liên hệ Zendesk.** Long-context vẫn hữu ích ở *quy mô nhỏ hơn*: đưa **toàn bộ thread của ticket hiện tại** (kể cả quoted reply dài) và **toàn bộ một bài Help Center** đã được chọn, thay vì chặt nhỏ chúng. Tức là long-context là công cụ để *tăng kích thước đơn vị truy xuất*, không phải để *thay thế truy xuất*.

---

## 4. So sánh các chiến lược: prompt-only, RAG, fine-tune, long-context, CAG

### 4.1 Năm chiến lược

1. **Prompt-only:** dùng tri thức tham số + chỉ dẫn trong system prompt. Rẻ, nhanh, nhưng chỉ đúng với tri thức phổ thông.
2. **RAG (Retrieval-Augmented Generation):** truy xuất một tập nhỏ tài liệu liên quan từ kho ngoài tại thời điểm hỏi, đưa vào context, yêu cầu model trả lời dựa trên đó.
3. **Fine-tune:** cập nhật trọng số (thường bằng LoRA) trên dữ liệu của bạn.
4. **Long-context:** đưa phần lớn hoặc toàn bộ kho tri thức vào prompt.
5. **CAG — Cache-Augmented Generation** (Chan et al., 2024): khi kho tri thức đủ nhỏ để vừa context, *tính trước* KV cache của toàn bộ kho một lần, lưu lại, rồi mỗi câu hỏi chỉ cần nối câu hỏi vào sau cache — không có bước truy xuất, không có lỗi truy xuất, độ trễ thấp hơn long-context thô vì không phải prefill lại. Về bản chất là long-context + prefix caching được đẩy tới cực hạn; prompt caching của các API thương mại là một dạng CAG "được quản lý".

### 4.2 Bằng chứng nghiên cứu

- **RAG vs fine-tune cho việc nạp tri thức:** Ovadia et al. (2024) so sánh fine-tune không giám sát và RAG trên các tác vụ kiến thức (bao gồm sự kiện *mới* sau cutoff); RAG nhất quán tốt hơn, và fine-tune gặp khó khi học sự kiện mới, cần nhiều biến thể diễn đạt của cùng một sự kiện mới cải thiện được. Kết hợp với Gekhman et al. (2024) ở mục 1.2: **fine-tune là công cụ cho hành vi/văn phong/định dạng, không phải cho sự kiện thay đổi.**
- **RAG vs long-context:** Li et al. (2024) so sánh trên nhiều benchmark: model long-context mạnh, khi đủ tài nguyên, đạt chất lượng trung bình cao hơn RAG; nhưng RAG rẻ hơn rất nhiều. Họ đề xuất **Self-Route**: model tự đánh giá câu hỏi có trả lời được từ các chunk RAG không; nếu không thì mới chuyển sang long-context. Cách này giữ chất lượng gần long-context với chi phí giảm mạnh. Bài học thiết kế: **định tuyến** giữa các chiến lược theo từng câu hỏi tốt hơn chọn một chiến lược cho tất cả.
- **CAG:** Chan et al. (2024) cho thấy với kho tri thức nhỏ, có giới hạn, CAG có thể ngang hoặc hơn RAG mà đơn giản hơn và nhanh hơn. Điều kiện tiên quyết chính là "kho vừa context và ít thay đổi".

### 4.3 Ma trận quyết định

Đánh giá theo nhu cầu của bài toán CS (++ rất phù hợp, + phù hợp, − kém, −− rất kém):

| Tiêu chí | Prompt-only | RAG | Fine-tune | Long-context | CAG |
|---|---|---|---|---|---|
| Tri thức riêng, thay đổi hằng tuần | −− | ++ (re-index) | −− (train lại) | + (đổi prompt) | − (tính lại cache) |
| Kho tri thức lớn (≫ context window) | −− | ++ | − | −− | −− |
| Trích dẫn nguồn / kiểm chứng được | −− | ++ | −− | + | + |
| Phân quyền theo tenant / ACL | −− | ++ (lọc khi truy xuất) | −− | − (phải tách prompt theo tenant) | − (cache theo tenant) |
| Chi phí mỗi request | ++ | + | ++ (khi đã train) | −− | + (sau khi có cache) |
| Độ trễ | ++ | + (thêm bước truy xuất) | ++ | −− | + |
| Văn phong, định dạng, keigo | + (few-shot) | + (few-shot từ macro) | ++ | + | + |
| Độ phức tạp vận hành | ++ | − (pipeline index) | − (pipeline train/eval) | + | − (quản lý cache) |
| Rủi ro lỗi do truy xuất sai | không có | có (cần đo) | không có | thấp | không có |

<!-- fig:strategy-matrix -->
<figure markdown="span">
  ![Ma trận quyết định của mục 4](assets/figures/02/strategy-matrix.light.svg#only-light){ loading=lazy }
  ![Ma trận quyết định của mục 4](assets/figures/02/strategy-matrix.dark.svg#only-dark){ loading=lazy }
  <figcaption>Hình 2.6 — Ma trận quyết định của mục 4.3 dưới dạng bản đồ màu (xanh: phù hợp, đỏ: kém); cột RAG được viền đậm.</figcaption>
</figure>
<!-- /fig -->

**Đọc ma trận cho Zendesk.**

- Tri thức sản phẩm + chính sách + ticket lịch sử → **RAG** là trục chính (lớn, thay đổi, cần trích dẫn và phân quyền).
- Văn phong email CS đa ngôn ngữ → bắt đầu bằng **few-shot** từ macro; chỉ **fine-tune** (LoRA/DPO) khi số liệu cho thấy tỉ lệ agent sửa văn phong còn cao (Module 09).
- System prompt + chính sách chung *nhỏ, ổn định* (vài nghìn token) → đặt ở đầu prompt, tận dụng **prompt caching** (một dạng CAG nhỏ).
- Thread dài của một ticket → **long-context** cục bộ.
- Dữ liệu tài khoản khách → **tool/API**, không phải chiến lược nào trong năm cái trên.

```mermaid
flowchart TD
    Q["Loại tri thức cần cho câu trả lời?"] --> A{"Thay đổi thường xuyên<br/>hoặc lớn hơn context?"}
    A -->|Có| R["RAG<br/>(+ lọc theo tenant/ACL)"]
    A -->|Không| B{"Nhỏ, ổn định,<br/>dùng ở mọi request?"}
    B -->|Có| C["Đặt đầu prompt + prompt caching<br/>(CAG thu nhỏ)"]
    B -->|Không| D{"Là hành vi/văn phong<br/>chứ không phải sự kiện?"}
    D -->|Có| F["Few-shot → nếu chưa đủ thì fine-tune (Module 09)"]
    D -->|Không| T{"Là dữ liệu theo khách hàng<br/>(tài khoản, đơn, hóa đơn)?"}
    T -->|Có| API["Tool / API call có kiểm tra quyền (Module 08)"]
    T -->|Không| R
```

---

## 5. Hình thức hóa RAG

Đến đây ta đã có lý do để *truy xuất*. Phần này mô tả RAG bằng ngôn ngữ xác suất, để bạn hiểu vì sao hệ thống được thiết kế như vậy và các biến thể khác nhau ở đâu.

### 5.1 Ký hiệu chung

- $x$: đầu vào (email khách, hoặc câu hỏi đã được viết lại).
- $y = (y_1, \dots, y_m)$: đầu ra (câu trả lời), gồm $m$ token.
- $\mathcal{Z}$: kho tài liệu (các chunk); $z \in \mathcal{Z}$ là một tài liệu.
- **Retriever** $p_\eta(z \mid x)$, tham số $\eta$. Với dense retriever: encoder query $\mathbf{q}(x) = \mathrm{Enc}_Q(x)$, encoder tài liệu $\mathbf{d}(z) = \mathrm{Enc}_D(z)$, và

$$
p_\eta(z \mid x) = \frac{\exp\big(\mathbf{d}(z)^\top \mathbf{q}(x)\big)}{\sum_{z' \in \mathcal{Z}} \exp\big(\mathbf{d}(z')^\top \mathbf{q}(x)\big)}.
$$

Trong thực tế tổng ở mẫu số chạy trên hàng triệu tài liệu là không khả thi, nên ta xấp xỉ bằng cách chỉ lấy **top-$k$** tài liệu có điểm cao nhất (tìm bằng ANN, Module 05) và chuẩn hóa softmax trên $k$ tài liệu đó.

- **Generator** $p_\theta(y \mid x, z)$: một model seq2seq hoặc decoder-only, sinh $y$ khi được cho $x$ và $z$.

Chú ý ý nghĩa của tích $\mathbf{d}(z)^\top\mathbf{q}(x)$: đúng là phép "tích vô hướng lớn nhất" ta gặp ở LM head (Module 01, mục 2.2) và ở attention, nhưng trên quy mô kho tài liệu. Cách học các encoder này là chủ đề của Module 03.

### 5.2 RAG gốc của Lewis et al. (2020): retrieval là biến ẩn

Lewis et al. (2020) kết hợp retriever DPR (encoder BERT) với generator BART, huấn luyện **end-to-end** trên cặp $(x, y)$ mà không cần nhãn "tài liệu nào đúng". Mẹo là coi tài liệu $z$ là **biến ẩn** và lấy biên (marginalize) trên nó. Hai cách lấy biên cho hai biến thể:

**RAG-Sequence** — cùng một tài liệu cho cả câu trả lời:

$$
p_{\text{RAG-Seq}}(y \mid x) \approx \sum_{z \in \text{top-}k(p_\eta(\cdot\mid x))} p_\eta(z \mid x)\, \underbrace{\prod_{i=1}^{m} p_\theta(y_i \mid x, z, y_{<i})}_{p_\theta(y \mid x, z)}.
$$

**RAG-Token** — mỗi token có thể dựa vào một tài liệu khác:

$$
p_{\text{RAG-Tok}}(y \mid x) \approx \prod_{i=1}^{m} \sum_{z \in \text{top-}k(p_\eta(\cdot\mid x))} p_\eta(z \mid x)\, p_\theta(y_i \mid x, z, y_{<i}).
$$

Sự khác biệt chỉ là **thứ tự giữa tổng và tích**: RAG-Sequence là "hỗn hợp các chuỗi" (mixture of sequences), RAG-Token là "hỗn hợp ở từng bước" (mixture per token).

**Huấn luyện.** Cực tiểu negative marginal log-likelihood

$$
\mathcal{L}(\theta, \eta) = -\sum_{(x,y)} \log p_{\text{RAG}}(y \mid x),
$$

gradient chảy về cả generator và encoder query (Lewis et al. giữ cố định encoder tài liệu để không phải index lại kho sau mỗi bước cập nhật — một chi tiết kỹ thuật rất thực tế).

**Giải mã.** RAG-Token giải mã như một LM bình thường vì mỗi bước có phân phối token $p'(y_i \mid x, y_{<i}) = \sum_z p_\eta(z|x)p_\theta(y_i|x,z,y_{<i})$, dùng được beam search chuẩn. RAG-Sequence không phân rã theo token, nên phải chạy beam search riêng cho từng tài liệu rồi cộng xác suất các giả thuyết (có thể cần thêm forward pass — gọi là "thorough decoding"; bản xấp xỉ rẻ hơn gọi là "fast decoding").

### 5.3 Ví dụ tính tay: RAG-Sequence vs RAG-Token

Email: *"Gói Pro cho tối đa bao nhiêu user?"* Retriever trả về $k = 3$ chunk với điểm $\mathbf{d}^\top\mathbf{q} = (2.0, 1.0, 0.5)$:

- $z_1$: bảng giá hiện hành (đúng),
- $z_2$: bài blog cũ năm 2023 về gói Pro (lỗi thời),
- $z_3$: bài hướng dẫn mời thành viên (liên quan một phần).

**Bước 1 — phân phối retriever** trên top-3: $\mathrm{softmax}(2.0, 1.0, 0.5)$. $e^{2} = 7.389$, $e^{1} = 2.718$, $e^{0.5} = 1.649$, tổng $11.756$ → $p_\eta(z \mid x) = (0.629, 0.231, 0.140)$.

**Bước 2 — giả sử câu trả lời đích có 2 token** $y = (y_1, y_2)$ = ("10", "user"). Xác suất generator cho từng tài liệu:

| | $p_\theta(y_1 \mid x, z)$ | $p_\theta(y_2 \mid x, z, y_1)$ | $p_\theta(y \mid x, z)$ |
|---|---|---|---|
| $z_1$ | 0,9 | 0,8 | 0,72 |
| $z_2$ | 0,5 | 0,1 | 0,05 |
| $z_3$ | 0,1 | 0,3 | 0,03 |

**RAG-Sequence:** $0.629 \times 0.72 + 0.231 \times 0.05 + 0.140 \times 0.03 = 0.4529 + 0.0116 + 0.0042 = 0.468$.

**RAG-Token:**

- Bước 1: $\sum_z p_\eta(z|x)\,p_\theta(y_1|x,z) = 0.629(0.9) + 0.231(0.5) + 0.140(0.1) = 0.566 + 0.116 + 0.014 = 0.696$.
- Bước 2: $0.629(0.8) + 0.231(0.1) + 0.140(0.3) = 0.503 + 0.023 + 0.042 = 0.568$.
- Tích: $0.696 \times 0.568 = 0.395$.

<!-- fig:rag-seq-vs-token -->
<figure markdown="span">
  ![Ví dụ tính tay của mục 5](assets/figures/02/rag-seq-vs-token.light.svg#only-light){ loading=lazy }
  ![Ví dụ tính tay của mục 5](assets/figures/02/rag-seq-vs-token.dark.svg#only-dark){ loading=lazy }
  <figcaption>Hình 2.7 — Ví dụ tính tay của mục 5.3: phân phối retriever, xác suất generator theo từng tài liệu, và xác suất câu trả lời dưới hai cách lấy biên.</figcaption>
</figure>
<!-- /fig -->

Ở ví dụ này RAG-Sequence gán xác suất cao hơn cho câu trả lời "nhất quán theo một nguồn". RAG-Token linh hoạt hơn (có thể lấy token này từ $z_1$, token kia từ $z_3$) — hữu ích khi câu trả lời cần *tổng hợp* nhiều tài liệu, nhưng cũng là con đường để **trộn lẫn thông tin từ nguồn lỗi thời** vào câu trả lời. Lewis et al. báo cáo hai biến thể mạnh ở các tác vụ khác nhau (ví dụ RAG-Token tốt trong một số tác vụ sinh câu hỏi cần ghép nhiều mẩu thông tin, RAG-Sequence tốt ở các tác vụ khác); không có biến thể thắng tuyệt đối.

**Bài học cho LLM hiện đại.** Ngày nay ta hiếm khi lấy biên tường minh như vậy. Pipeline phổ biến là **nối $k$ chunk vào một prompt** và để attention của LLM tự "lấy biên mềm" bên trong (giống RAG-Token nhưng do attention quyết định, không phải $p_\eta$). Hệ quả: điểm retriever $p_\eta(z|x)$ không còn được dùng trực tiếp để trọng số hóa — thứ tự và cách trình bày chunk trong prompt trở thành "trọng số" ngầm (nhớ lost in the middle). Đó là lý do reranking và sắp xếp context có vai trò lớn (Module 06, 07).

### 5.4 REALM: retrieval-augmented pretraining và gradient trọng số theo posterior

**REALM** (Guu et al., 2020) đi trước RAG vài tháng, đưa retriever vào ngay từ **pretraining** (masked language modeling): để điền token bị che trong một câu, model truy xuất tài liệu từ Wikipedia rồi dự đoán. Mô hình xác suất cùng dạng:

$$
p(y \mid x) = \sum_{z \in \mathcal{Z}} p(y \mid z, x)\, p(z \mid x),
$$

trong đó $y$ là token bị che. Phần thú vị là **gradient của retriever**. Đặt $f(x,z) = \mathbf{d}(z)^\top\mathbf{q}(x)$ là điểm retriever. Có thể chứng minh:

$$
\nabla \log p(y \mid x) = \sum_{z} \Big[\,p(z \mid x, y) - p(z \mid x)\,\Big]\, \nabla f(x, z) \;+\; (\text{số hạng theo generator}),
$$

trong đó $p(z \mid x, y) = \dfrac{p(y\mid z,x)\,p(z\mid x)}{\sum_{z'} p(y\mid z',x)\,p(z'\mid x)}$ là **posterior** của tài liệu sau khi đã biết đáp án.

**Dẫn xuất ngắn** (chỉ phần retriever, coi $p(y|z,x)$ cố định). Đặt $p(y|x) = \sum_z p(y|z,x)p(z|x)$ với $p(z|x) = e^{f(z)}/\sum_{z'}e^{f(z')}$. Ta có $\partial p(z|x)/\partial f(z'') = p(z|x)\big(\mathbb 1[z=z''] - p(z''|x)\big)$. Do đó

$$
\frac{\partial \log p(y|x)}{\partial f(z'')} = \frac{1}{p(y|x)}\sum_z p(y|z,x)\,p(z|x)\big(\mathbb 1[z=z''] - p(z''|x)\big) = p(z''|x,y) - p(z''|x).
$$

**Trực giác.** Điểm của tài liệu $z$ được **tăng** nếu *sau khi biết đáp án*, $z$ trở nên đáng tin hơn so với trước ($p(z|x,y) > p(z|x)$) — tức $z$ thực sự giúp dự đoán đúng; và **giảm** nếu ngược lại. Retriever học được "tài liệu nào hữu ích" mà **không cần nhãn tài liệu**.

**Ví dụ số** (dùng lại 5.3, coi cả chuỗi $y$ là "đáp án"). $p(z|x) = (0.629, 0.231, 0.140)$, $p(y|z,x) = (0.72, 0.05, 0.03)$ → $p(y|x) = 0.468$ và

$$
p(z\mid x,y) = \frac{(0.4529,\ 0.0116,\ 0.0042)}{0.4687} = (0.966,\ 0.025,\ 0.009).
$$

Gradient theo điểm: $(0.966 - 0.629,\ 0.025 - 0.231,\ 0.009 - 0.140) = (+0.337, -0.206, -0.131)$. Retriever được đẩy để xếp bảng giá hiện hành cao hơn nữa, và xếp bài blog lỗi thời thấp đi — đúng điều ta muốn.

<!-- fig:realm-posterior -->
<figure markdown="span">
  ![Ví dụ số của mục 5](assets/figures/02/realm-posterior.light.svg#only-light){ loading=lazy }
  ![Ví dụ số của mục 5](assets/figures/02/realm-posterior.dark.svg#only-dark){ loading=lazy }
  <figcaption>Hình 2.8 — Ví dụ số của mục 5.4: gradient theo điểm retriever bằng posterior trừ prior, đẩy bảng giá hiện hành lên và bài blog cũ xuống.</figcaption>
</figure>
<!-- /fig -->

**Chi phí thực tế.** Vì encoder tài liệu thay đổi khi huấn luyện, REALM phải **index lại kho** (tính lại embedding hàng triệu tài liệu) định kỳ theo kiểu bất đồng bộ. Đây là lý do nhiều hệ thống (kể cả RAG gốc) đóng băng encoder tài liệu.

> **Liên hệ Zendesk.** Ý tưởng posterior của REALM có một phiên bản rất thực dụng: với 200.000 ticket đã giải quyết, ta biết **câu trả lời cuối cùng** agent đã gửi ($y$). Tài liệu Help Center nào làm câu trả lời đó "dễ sinh" nhất (ví dụ được agent đính link, hoặc có độ tương đồng cao với câu trả lời) là tài liệu dương tính tốt cho email đó — một nguồn nhãn yếu để **đánh giá retriever** (Module 10) và **fine-tune embedding** (Module 09) mà không cần gán nhãn tay.

### 5.5 Fusion-in-Decoder (FiD)

Izacard & Grave (2021) đề xuất một cách "lấy biên" khác cho model encoder–decoder (T5):

1. **Encode độc lập** từng cặp (câu hỏi, tài liệu $z_j$), $j = 1..k$, bằng encoder → $k$ chuỗi trạng thái ẩn.
2. **Nối** tất cả trạng thái ẩn lại và để **decoder** cross-attend lên toàn bộ.

$$
H = \big[\mathrm{Enc}(x \oplus z_1);\ \mathrm{Enc}(x \oplus z_2);\ \dots;\ \mathrm{Enc}(x \oplus z_k)\big], \qquad y_i \sim p_\theta(\cdot \mid H, y_{<i}).
$$

**Vì sao hiệu quả.** Self-attention trong encoder có chi phí bậc hai theo độ dài. Nếu nối $k$ tài liệu dài $\ell$ token rồi mới encode: $O\big((k\ell)^2\big)$. FiD encode riêng: $O(k\ell^2)$. Ví dụ $k = 100$, $\ell = 250$: $(k\ell)^2 = 6.25\cdot10^{8}$ so với $k\ell^2 = 6.25\cdot10^{6}$ — rẻ hơn 100 lần, cho phép dùng tới 100 đoạn văn. Izacard & Grave quan sát chất lượng tăng đều khi tăng số đoạn văn — bằng chứng sớm rằng **decoder tổng hợp được thông tin từ nhiều nguồn tốt hơn việc chọn một nguồn**.

<!-- fig:fid-cost -->
<figure markdown="span">
  ![Fusion-in-Decoder encode từng cặp (câu hỏi, tài liệu) riêng nên chỉ tính các khối trên đường chéo của ma trận attention](assets/figures/02/fid-cost.light.svg#only-light){ loading=lazy }
  ![Fusion-in-Decoder encode từng cặp (câu hỏi, tài liệu) riêng nên chỉ tính các khối trên đường chéo của ma trận attention](assets/figures/02/fid-cost.dark.svg#only-dark){ loading=lazy }
  <figcaption>Hình 2.9 — Fusion-in-Decoder encode từng cặp (câu hỏi, tài liệu) riêng nên chỉ tính các khối trên đường chéo của ma trận attention.</figcaption>
</figure>
<!-- /fig -->

**Liên hệ hiện nay.** Kiến trúc decoder-only không tách được encoder như vậy, nhưng ý tưởng "xử lý tài liệu độc lập rồi hợp nhất" xuất hiện lại dưới dạng: tính KV cache riêng cho từng chunk và tái sử dụng (một hướng nghiên cứu của CAG/prefix caching), hoặc map–reduce (tóm tắt từng tài liệu rồi tổng hợp, Module 08).

### 5.6 RETRO và Atlas (mức ý tưởng)

**RETRO** (Borgeaud et al., 2022 — DeepMind): đưa retrieval vào *bên trong* kiến trúc LM ở quy mô rất lớn. Chuỗi đầu vào được chia thành các khúc 64 token; với mỗi khúc, truy xuất các láng giềng gần nhất (theo embedding BERT đóng băng) từ một kho lên tới **hàng nghìn tỷ token**, và các lớp **chunked cross-attention** cho phép khúc tiếp theo attend tới láng giềng của khúc trước. Thông điệp chính: một model với kho truy xuất khổng lồ có thể đạt hiệu năng ngang model *lớn hơn khoảng 25 lần* tham số trên benchmark mô hình hóa ngôn ngữ — tức **chuyển một phần "tri thức" từ tham số sang kho ngoài**. Kho có thể cập nhật mà không train lại (dù chất lượng còn phụ thuộc mức chồng lấp giữa dữ liệu test và kho).

**Atlas** (Izacard et al., 2023, JMLR; preprint 2022): retriever Contriever + generator T5 kiểu FiD, **huấn luyện chung (jointly)** retriever và generator, nghiên cứu nhiều loss để huấn luyện retriever từ tín hiệu của generator (ví dụ chưng cất attention của decoder, hoặc posterior kiểu EMDR²/perplexity). Kết quả nổi bật: trên Natural Questions chỉ với **64 ví dụ huấn luyện**, Atlas 11B đạt độ chính xác trên 42%, vượt PaLM 540B dù ít tham số hơn khoảng 50 lần. Thông điệp: với tác vụ cần nhiều tri thức, **truy xuất tốt quan trọng hơn kích thước model**.

| | RAG (Lewis) | REALM | FiD | RETRO | Atlas |
|---|---|---|---|---|---|
| Năm | 2020 | 2020 | 2021 | 2022 | 2022/2023 |
| Generator | BART | BERT-style MLM | T5 enc–dec | Decoder + chunked cross-attn | T5 (FiD) |
| Truy xuất vào lúc | Fine-tune & suy luận | Cả pretraining | Fine-tune & suy luận | Pretraining & suy luận | Pretraining & fine-tune |
| Cách hợp nhất tài liệu | Lấy biên xác suất | Lấy biên xác suất | Decoder attend lên mọi tài liệu | Cross-attention theo khúc | Như FiD |
| Huấn luyện retriever | Encoder query (end-to-end) | End-to-end + re-index | Không (retriever riêng) | Không (BERT đóng băng) | Chung với generator |
| Ý nghĩa với kỹ sư 2026 | Khung xác suất cơ bản | Retrieval như biến ẩn; nhãn yếu từ posterior | Tổng hợp nhiều nguồn | Tri thức ở kho thay vì tham số | Retrieval tốt > model lớn |

**Điều gì thay đổi khi chuyển sang LLM 2023–2026?** Hầu hết hệ thống RAG production không huấn luyện chung retriever và generator; chúng dùng một **LLM có sẵn (đóng băng hoặc fine-tune nhẹ)** + **retriever có sẵn** (embedding + BM25), ghép qua prompt. Lý do: LLM đủ mạnh trong việc đọc context; huấn luyện chung quá đắt và khó với model hàng chục tỷ tham số. Nhưng các ý tưởng trên vẫn sống: lấy biên → hybrid/multi-query fusion; posterior → nhãn yếu cho đánh giá; FiD → tổng hợp nhiều nguồn; RETRO/Atlas → đầu tư vào retrieval có lợi hơn đầu tư vào model to.

---

## 6. Pipeline RAG "naive" chuẩn và bản đồ điểm hỏng

### 6.1 Ba giai đoạn

```mermaid
flowchart LR
    subgraph OFF["Offline: Indexing"]
        S["Nguồn: Help Center, macro,<br/>ticket đã giải quyết, chính sách"] --> P["Làm sạch + chuẩn hóa<br/>(NFC, bỏ chữ ký, quoted reply)"]
        P --> C["Chunking + metadata<br/>(product, lang, updated_at, tenant)"]
        C --> EM["Embedding (+ index BM25)"]
        EM --> IDX[("Vector DB / search index")]
    end
    subgraph ON["Online: mỗi ticket"]
        X["Email khách"] --> QE["Biểu diễn query"]
        QE --> RET["Retrieve top-k<br/>(+ lọc tenant/ACL)"]
        IDX --> RET
        RET --> PR["Ghép prompt:<br/>system + chunk có ID + email"]
        PR --> LLM["LLM sinh trả lời"]
        LLM --> OUT["Draft / internal note"]
    end
```

Hình thức hóa ngắn gọn của pipeline naive dùng trong production (khác RAG gốc ở chỗ *không lấy biên*, chỉ ghép prompt):

$$
Z_k(x) = \mathrm{TopK}_{z \in \mathcal{Z},\ \mathrm{filter}(z,\, x)}\ s(x, z), \qquad \hat{y} = \mathrm{Decode}\Big(p_\theta\big(\cdot \mid \mathrm{Prompt}(x, Z_k(x))\big)\Big),
$$

với $s(x,z)$ là hàm điểm (dense, BM25, hoặc hybrid), $\mathrm{filter}$ là điều kiện metadata (tenant, ngôn ngữ, sản phẩm, hiệu lực), $\mathrm{Prompt}$ là template ghép.

**Đúng của câu trả lời phân rã thành các sự kiện nối tiếp.** Nếu ta viết

$$
P(\text{đúng}) = P(\text{kho có đáp án}) \cdot P(\text{đáp án trong top-}k \mid \text{có}) \cdot P(\text{LLM dùng đúng} \mid \text{trong top-}k) + P(\text{đúng} \mid \text{không có trong top-}k)\cdot(\dots)
$$

và giả định số hạng cuối xấp xỉ 0 (với chính sách nội bộ, model không thể đúng nếu không có tài liệu — trừ khi đoán trúng), thì với các giá trị minh họa $0.95 \times 0.85 \times 0.90 \approx 0.73$. **Ba con số khá cao nhân lại cho ra một con số khá thấp.** Đây là lý do mỗi khâu phải được đo riêng (Module 10): nếu chỉ đo đầu ra cuối, bạn không biết sửa khâu nào.

<!-- fig:pipeline-chain -->
<figure markdown="span">
  ![Xác suất trả lời đúng của pipeline naive là tích xác suất các khâu (giá trị minh họa của mục 6](assets/figures/02/pipeline-chain.light.svg#only-light){ loading=lazy }
  ![Xác suất trả lời đúng của pipeline naive là tích xác suất các khâu (giá trị minh họa của mục 6](assets/figures/02/pipeline-chain.dark.svg#only-dark){ loading=lazy }
  <figcaption>Hình 2.10 — Xác suất trả lời đúng của pipeline naive là tích xác suất các khâu (giá trị minh họa của mục 6.1).</figcaption>
</figure>
<!-- /fig -->

### 6.2 Bảy điểm hỏng (Barnett et al., 2024)

Barnett et al. rút kinh nghiệm từ ba hệ thống RAG thực tế (trong các lĩnh vực nghiên cứu, giáo dục, y sinh) và đề xuất bảy điểm hỏng (failure point, FP). Mình ánh xạ từng điểm vào Zendesk và module xử lý:

| FP | Tên | Nghĩa | Ví dụ trong Zendesk | Xử lý ở |
|---|---|---|---|---|
| FP1 | Missing content | Kho không có đáp án; lý tưởng là hệ thống nói "không biết", nhưng thường lại trả lời bịa | Khách hỏi tính năng chưa có tài liệu; AI bịa cách cấu hình | Module 04 (độ phủ kho), Module 07 (abstention), Module 10 (escalate) |
| FP2 | Missed the top ranked documents | Có đáp án trong kho nhưng không lọt top-$k$ | Email tiếng Nhật, bài Help Center chỉ có tiếng Anh; embedding không khớp xuyên ngôn ngữ | Module 03, 05 (hybrid, đa ngữ), Module 06 (rewrite) |
| FP3 | Not in context | Được truy xuất nhưng bị loại khi ghép context (giới hạn token, chiến lược hợp nhất) | Reranker/cắt context bỏ mất chunk chứa điều kiện hoàn tiền | Module 06 (rerank, nén context) |
| FP4 | Not extracted | Đáp án có trong context nhưng LLM không trích ra được (nhiễu, mâu thuẫn, lost in the middle) | Hai phiên bản chính sách cùng xuất hiện, model lấy bản cũ | Module 04 (versioning), Module 07 (thứ tự, xử lý mâu thuẫn) |
| FP5 | Wrong format | Sai định dạng yêu cầu (bảng, JSON, danh sách) | JSON quyết định route bị hỏng; email thiếu lời chào keigo | Module 07 (structured output, văn phong) |
| FP6 | Incorrect specificity | Câu trả lời quá chung chung hoặc quá chi tiết so với nhu cầu | Khách hỏi "vì sao bị trừ tiền 2 lần?", AI dán nguyên chính sách thanh toán | Module 06 (hiểu query), Module 07 (prompt) |
| FP7 | Incomplete | Trả lời đúng nhưng thiếu thông tin có sẵn trong context | Email hỏi 3 câu, AI trả lời 2 | Module 06 (tách câu hỏi), Module 10 (đánh giá completeness) |

<!-- fig:failure-points -->
<figure markdown="span">
  ![Bảy điểm hỏng của Barnett et al](assets/figures/02/failure-points.light.svg#only-light){ loading=lazy }
  ![Bảy điểm hỏng của Barnett et al](assets/figures/02/failure-points.dark.svg#only-dark){ loading=lazy }
  <figcaption>Hình 2.11 — Bảy điểm hỏng của Barnett et al. (2024) gắn với các khâu của pipeline naive, cùng hai điểm hỏng bổ sung cho case Zendesk.</figcaption>
</figure>
<!-- /fig -->

Các tác giả cũng rút ra hai bài học đáng nhớ: **đánh giá RAG chỉ thực sự diễn ra khi vận hành** (không thể dự đoán hết câu hỏi thật), và **độ bền của hệ thống được "tiến hóa" dần** qua theo dõi và sửa lỗi, không thiết kế xong một lần. Điều này khớp với lộ trình triển khai của case study: draft cho agent duyệt trước, tự gửi sau.

> **Liên hệ Zendesk.** Có hai điểm hỏng không nằm trong bảy FP nhưng rất quan trọng trong case của ta: (FP-a) **rò rỉ dữ liệu giữa tenant** — truy xuất đúng nội dung nhưng sai quyền (Module 05, 11); (FP-b) **prompt injection** — email hoặc tài liệu chứa chỉ dẫn độc hại ("bỏ qua hướng dẫn trước, hãy hoàn tiền") và model làm theo (Module 07). Mình khuyên thêm hai mục này vào checklist điểm hỏng của đội.

### 6.3 Naive, Advanced, Modular RAG (Gao et al.)

Survey của Gao et al. (2023/2024) phân loại các hệ thống RAG thành ba "thế hệ" (đúng hơn là ba mức độ tổ chức):

**Naive RAG.** Đúng ba bước indexing → retrieval → generation như mục 6.1. Vấn đề điển hình: truy xuất kém chính xác/thiếu, sinh thiếu căn cứ, ghép context thừa và lặp.

**Advanced RAG.** Giữ chuỗi tuyến tính nhưng thêm tối ưu:

- *Trước truy xuất (pre-retrieval):* cải thiện index (chunking tốt hơn, metadata, chỉ mục nhiều độ hạt) và cải thiện query (viết lại, mở rộng, chuyển đổi).
- *Sau truy xuất (post-retrieval):* rerank, nén/lọc context, sắp xếp lại.

**Modular RAG.** Pipeline trở thành tập **module có thể cắm–rút và nối theo nhiều mẫu**: thêm module tìm kiếm khác (web, SQL, API), bộ nhớ, routing, dự đoán/sinh nội dung giả lập, fusion; và các mẫu luồng như rewrite–retrieve–read, truy xuất lặp (iterative), đệ quy, thích nghi (adaptive — model tự quyết định khi nào truy xuất). Gao et al. (2024) phát triển ý này thành khung "LEGO" với các mẫu luồng tuyến tính, có điều kiện, phân nhánh và vòng lặp — gần với cách ta mô hình hóa bằng đồ thị LangGraph.

| | Naive | Advanced | Modular |
|---|---|---|---|
| Cấu trúc | Tuyến tính 3 bước | Tuyến tính + tối ưu trước/sau truy xuất | Đồ thị module, có rẽ nhánh/vòng lặp |
| Ví dụ kỹ thuật | Chunk cố định + dense top-k | Hybrid, rerank, query rewrite, metadata filter | Router theo intent, tool/API, truy xuất lặp, tự đánh giá |
| Khi nào đủ | Prototype, FAQ đơn giản | Đa số câu hỏi FAQ/hướng dẫn | Câu hỏi nhiều bước, cần dữ liệu tài khoản, cần quyết định hành động |
| Module khóa học | 03–05 | 04–07 | 08, 12 |

<!-- fig:rag-paradigms -->
<figure markdown="span">
  ![Naive, Advanced và Modular RAG theo Gao et al](assets/figures/02/rag-paradigms.light.svg#only-light){ loading=lazy }
  ![Naive, Advanced và Modular RAG theo Gao et al](assets/figures/02/rag-paradigms.dark.svg#only-dark){ loading=lazy }
  <figcaption>Hình 2.12 — Naive, Advanced và Modular RAG theo Gao et al.: Advanced thêm bước trước và sau truy xuất; Modular có router, nhiều nguồn và vòng lặp.</figcaption>
</figure>
<!-- /fig -->

Mình khuyên: **bắt đầu ở Advanced RAG có đo lường**, chỉ chuyển dần sang Modular cho những luồng mà số liệu cho thấy cần. Module 08 bàn kỹ "khi nào phức tạp hóa là xứng đáng".

---

## 7. Áp vào Zendesk: vì sao đây là RAG + phân loại + quyết định escalate

Đọc lại hai yêu cầu của ticket: (1) **AI trả lời email cho khách qua Zendesk**; (2) **AI thông báo CS vào xử lý khi khách muốn gặp người hoặc khi AI tự đánh giá cần người can thiệp**. Yêu cầu (1) cần RAG. Yêu cầu (2) không phải bài toán sinh văn bản — nó là bài toán **phân loại và ra quyết định dưới bất định**. Hệ thống đúng phải có cả ba thành phần.

### 7.1 Ba bài toán con

**(A) Phân loại (trước khi sinh).** Với mỗi email $x$, dự đoán:

- ngôn ngữ (vi/en/ja/lẫn),
- intent (đăng nhập, hóa đơn, hoàn tiền, lỗi API, góp ý...),
- độ nhạy: có liên quan giá/hoàn tiền/pháp lý/bảo mật không,
- **tín hiệu "muốn gặp người"** (rõ ràng: "cho tôi gặp nhân viên"; ngầm: bức xúc, đe dọa hủy hợp đồng),
- dấu hiệu prompt injection / spam.

Các nhãn này quyết định *có nên* chạy RAG không, chạy với bộ lọc nào (sản phẩm, ngôn ngữ), và các ngưỡng nào áp dụng.

**(B) RAG (sinh có căn cứ).** Truy xuất từ Help Center, macro, ticket lịch sử, chính sách (đúng tenant, đúng phiên bản), sinh câu trả lời **có trích dẫn**, cho phép abstention.

**(C) Quyết định (sau khi sinh).** Chọn một trong ba hành động $a \in \{\text{gửi},\ \text{draft},\ \text{escalate}\}$ dựa trên các tín hiệu: kết quả phân loại, điểm retrieval, kết quả kiểm tra faithfulness, độ tin cậy đã hiệu chuẩn. Dạng chuẩn của quyết định là tối thiểu hóa rủi ro kỳ vọng:

$$
a^* = \arg\min_{a} \sum_{s \in \{\text{đúng},\ \text{sai}\}} C(a, s)\, P(s \mid x, \hat{y}),
$$

với $C(a,s)$ là chi phí của hành động $a$ khi trạng thái thật là $s$ (ví dụ gửi email sai chính sách: chi phí rất lớn; escalate một email đúng: chi phí bằng thời gian một agent). Đây là phiên bản tổng quát của ngưỡng $p > c/(1+c)$ ở mục 1.2. Module 10 ước lượng $P(s|\cdot)$ và chọn ngưỡng; Module 12 lắp thành chính sách escalation.

```mermaid
flowchart TD
    E["Webhook: ticket mới / khách trả lời"] --> CL["(A) Phân loại:<br/>ngôn ngữ, intent, độ nhạy,<br/>muốn gặp người, injection"]
    CL -->|"Muốn gặp người / nhạy cao"| ESC["Escalate: internal note tóm tắt<br/>+ đổi group/tag + Slack/email"]
    CL -->|"Còn lại"| RAG["(B) RAG: retrieve (lọc tenant)<br/>→ sinh trả lời có trích dẫn"]
    RAG --> V["Kiểm tra: citation, faithfulness,<br/>guardrail chính sách"]
    V --> D{"(C) Quyết định<br/>theo rủi ro + ngưỡng"}
    D -->|"Tin cậy cao + intent rủi ro thấp<br/>(giai đoạn 2)"| SEND["Public reply"]
    D -->|"Trung bình"| DRAFT["Internal note (draft) cho agent duyệt"]
    D -->|"Thấp / abstain"| ESC
```

### 7.2 Vì sao không "chỉ RAG"

- Một hệ thống chỉ có (B) sẽ **luôn sinh ra câu trả lời** — đúng kiểu hành vi mà post-training và benchmark nhị phân khuyến khích (mục 1.2). Không có (A) và (C), không có điểm nào trong pipeline được *phép* nói "việc này cần người".
- Yêu cầu "khách muốn gặp người" phải được tôn trọng **kể cả khi RAG trả lời được** — đó là quyết định trải nghiệm khách hàng, không phải quyết định tri thức.
- Một số intent (khiếu nại thanh toán, yêu cầu xóa dữ liệu, sự cố bảo mật) cần escalate theo *quy tắc*, bất kể độ tin cậy.
- Lộ trình triển khai (draft → tự gửi intent rủi ro thấp → mở rộng) thực chất là **dịch chuyển ngưỡng của (C)** dựa trên số liệu; không có (C) tường minh thì không có lộ trình.

### 7.3 Chiến lược tri thức cho từng nguồn (tổng hợp module)

| Nguồn (giả định case study) | Quy mô | Tần suất đổi | Chiến lược | Ghi chú |
|---|---|---|---|---|
| Help Center | ~800 bài | Hằng tuần | RAG (hybrid + rerank) | Đơn vị truy xuất có thể là cả bài nếu ngắn (long-context cục bộ) |
| Macro CS | ~300 | Thỉnh thoảng | RAG + few-shot văn phong | Tách vai trò: văn phong vs căn cứ sự thật |
| Ticket đã giải quyết | ~200.000 | Tăng mỗi ngày | RAG có lọc (CSAT cao, đã che PII), dùng làm ví dụ/nhãn yếu | Không dùng làm "nguồn chính sách" |
| Chính sách giá/hoàn tiền/SLA | Nhỏ | Thay đổi có phiên bản | RAG từ một nguồn có thẩm quyền + guardrail; có thể luôn đưa vào prompt (CAG nhỏ) nếu ngắn | Ưu tiên `updated_at`, hiệu lực |
| System prompt, quy tắc chung | Vài nghìn token | Hiếm | Đầu prompt + prompt caching | |
| Dữ liệu tài khoản khách | Theo khách | Thời gian thực | Tool/API có kiểm tra quyền | Không index chung |
| Văn phong email (vi/en/ja) | — | Hiếm | Few-shot → (nếu cần) LoRA/DPO | Module 09 |

---

## Lỗi thường gặp & cách xử lý

| Triệu chứng | Nguyên nhân gốc | Cách xử lý |
|---|---|---|
| AI trả lời tự tin về tính năng/chính sách không có trong tài liệu | FP1 (missing content) + model được tối ưu để luôn trả lời | Cho phép và yêu cầu abstention trong prompt; kiểm tra citation; đo tỉ lệ "trả lời khi không có căn cứ" (Module 07, 10) |
| Trích chính sách hoàn tiền phiên bản cũ | Kho có nhiều phiên bản, không lọc theo hiệu lực; FP4 | Metadata `valid_from/valid_to`, `updated_at`; lọc trước khi truy xuất; quy tắc ưu tiên nguồn mới (Module 04, 07) |
| Tăng `k` từ 5 lên 30 nhưng chất lượng giảm | Context dài hơn → nhiễu, lost in the middle, context rot | Giữ $k$ nhỏ sau rerank; đặt chunk quan trọng ở đầu/cuối; đo theo $k$ (Module 06) |
| "Cứ dùng model 1M token cho đơn giản" làm hóa đơn API tăng vọt | Chi phí input tuyến tính (tiền) và bậc hai (tính toán) theo độ dài | Tính chi phí theo ticket trước khi chọn; dùng long-context cục bộ (thread, một bài) thay vì toàn kho |
| Fine-tune trên tài liệu sản phẩm nhưng model vẫn sai và còn bịa nhiều hơn | Fine-tune kém trong việc nạp sự kiện mới, có thể tăng hallucination | Đưa sự kiện vào RAG; chỉ fine-tune cho văn phong/hành vi (Module 09) |
| Email tiếng Nhật không tìm được bài Help Center tiếng Anh tương ứng | FP2: embedding/BM25 không khớp xuyên ngôn ngữ | Embedding đa ngữ, dịch/viết lại query, hybrid (Module 03, 05, 06) |
| Câu trả lời đúng nhưng bỏ sót câu hỏi thứ hai trong email | FP7 (incomplete): một email nhiều câu hỏi | Tách câu hỏi trước truy xuất; checklist trong prompt; đánh giá completeness (Module 06, 10) |
| Khách yêu cầu gặp người nhưng vẫn nhận email từ AI | Thiếu bước phân loại (A) chạy trước RAG | Bộ phân loại "muốn gặp người" chạy đầu tiên, quy tắc cứng → escalate |
| Chỉ có một con số "accuracy" cuối, không biết sửa đâu | Không đo theo từng khâu | Đo riêng: độ phủ kho, recall@k, faithfulness, quyết định (Module 10) |

---

## Tóm tắt (cheat-sheet)

**Giới hạn LLM**

- Hallucination: intrinsic (mâu thuẫn nguồn) vs extrinsic (không kiểm chứng được); factuality vs faithfulness (chỉ dẫn, context, tự nhất quán).
- Nguyên nhân: tri thức đuôi dài, mục tiêu cross-entropy không có "không biết", benchmark nhị phân thưởng việc đoán, decoding/context nhiễu. Fine-tune trên sự kiện mới có thể tăng bịa.
- Trả lời có lợi khi $p > c/(1+c)$; $c = 0$ ⇒ luôn đoán. Hệ thống CS cần $c$ lớn.
- Knowledge cutoff + tri thức riêng + phân quyền tenant ⇒ tri thức phải nằm ngoài tham số.
- Context dài: lost in the middle (hình chữ U), độ dài hiệu dụng < độ dài quảng cáo (RULER, context rot); chi phí prefill $\approx 2Nn + 2Ln^2d$.

**Chọn chiến lược**

- RAG cho tri thức lớn/thay đổi/cần trích dẫn/cần ACL; fine-tune cho hành vi/văn phong; long-context cục bộ cho thread/bài đơn; CAG/prompt caching cho phần nhỏ ổn định; tool/API cho dữ liệu khách.
- Định tuyến theo câu hỏi (Self-Route) tốt hơn chọn một chiến lược cho tất cả.

**Hình thức hóa RAG**

- Retriever $p_\eta(z|x) \propto \exp(\mathbf d(z)^\top\mathbf q(x))$ trên top-$k$.
- RAG-Sequence: $\sum_z p_\eta(z|x)\prod_i p_\theta(y_i|x,z,y_{<i})$. RAG-Token: $\prod_i\sum_z p_\eta(z|x)p_\theta(y_i|x,z,y_{<i})$.
- REALM: $\partial\log p(y|x)/\partial f(z) = p(z|x,y) - p(z|x)$ ⇒ học retriever không cần nhãn tài liệu.
- FiD: encode riêng từng tài liệu ($O(k\ell^2)$ thay vì $O(k^2\ell^2)$), decoder hợp nhất.
- RETRO: kho truy xuất nghìn tỷ token thay cho tham số. Atlas: retrieval tốt + few-shot thắng model lớn hơn ~50 lần.

**Pipeline & điểm hỏng**

- Naive: index → retrieve → generate. $P(\text{đúng}) \approx P(\text{có})\cdot P(\text{top-}k)\cdot P(\text{dùng đúng})$ ⇒ đo từng khâu.
- 7 FP: missing content, missed top-ranked, not in context, not extracted, wrong format, incorrect specificity, incomplete. Thêm: rò rỉ tenant, prompt injection.
- Naive → Advanced (pre/post-retrieval) → Modular (đồ thị module, routing, vòng lặp).

**Zendesk = (A) phân loại + (B) RAG + (C) quyết định** $a^* = \arg\min_a\sum_s C(a,s)P(s|\cdot)$.

---

## Câu hỏi tự kiểm tra / phỏng vấn

**1. Phân biệt intrinsic và extrinsic hallucination bằng một ví dụ trong email CS. Loại nào khó phát hiện tự động hơn?**

<details markdown="1"><summary>Gợi ý đáp án</summary>

Intrinsic: tài liệu nói 14 ngày, AI viết 30 ngày (mâu thuẫn). Extrinsic: tài liệu không nhắc phí, AI viết "miễn phí". Extrinsic thường khó hơn: không có câu nào trong nguồn mâu thuẫn trực tiếp, bộ kiểm tra phải xác định rằng *không có* căn cứ (NLI cho nhãn "neutral" thay vì "contradiction"), và nội dung có thể tình cờ đúng.

</details>

**2. Vì sao một model được tối ưu theo benchmark đúng/sai nhị phân có xu hướng đoán thay vì nói "không biết"? Tính ngưỡng trả lời khi phạt sai gấp 4 lần thưởng đúng.**

<details markdown="1"><summary>Gợi ý đáp án</summary>

Với thưởng $+1$ khi đúng, $0$ khi sai và $0$ khi từ chối, kỳ vọng điểm khi đoán là $p \ge 0$ — không bao giờ thua từ chối, nên tối ưu là luôn đoán. Phạt sai $c = 4$: trả lời khi $p - 4(1-p) > 0 \iff p > 0.8$.

</details>

**3. Nêu hai bằng chứng nghiên cứu cho thấy không nên dùng fine-tune để nạp chính sách hoàn tiền mới.**

<details markdown="1"><summary>Gợi ý đáp án</summary>

Ovadia et al. (2024): RAG nhất quán tốt hơn fine-tune không giám sát trong việc nạp tri thức, đặc biệt sự kiện mới. Gekhman et al. (2024): model học ví dụ chứa tri thức mới chậm, và khi đã học thì xu hướng hallucinate tăng. Thêm lý do vận hành: chính sách đổi hằng tuần, cần trích dẫn và phân quyền.

</details>

**4. Một đồng nghiệp đề xuất: "Model mới có 1M token context, cứ đưa cả 800 bài Help Center vào, bỏ RAG đi." Phản biện bằng số và bằng chất lượng.**

<details markdown="1"><summary>Gợi ý đáp án</summary>

Chi phí: ~640K token/lời gọi × ~5.250 lời gọi/ngày ≈ 3,4 tỷ token/ngày, gấp ~100 lần RAG (~6K token/lời gọi); prompt caching giảm còn khoảng 10 lần nhưng vẫn đắt hơn. Tính toán prefill tăng gần bậc hai. Chất lượng: lost in the middle, context rot, nhiều bài gần giống nhau là nhiễu mạnh. Không chứa được 200.000 ticket lịch sử. Không phân quyền theo tenant được. Phương án tốt hơn: RAG, dùng long-context cục bộ cho thread/bài đã chọn, hoặc Self-Route.

</details>

**5. Viết công thức RAG-Sequence và RAG-Token. Khi nào RAG-Token có lợi, khi nào nguy hiểm?**

<details markdown="1"><summary>Gợi ý đáp án</summary>

Seq: $\sum_z p_\eta(z|x)\prod_i p_\theta(y_i|x,z,y_{<i})$. Token: $\prod_i \sum_z p_\eta(z|x)p_\theta(y_i|x,z,y_{<i})$. Token có lợi khi câu trả lời cần ghép thông tin từ nhiều tài liệu. Nguy hiểm khi có tài liệu lỗi thời/mâu thuẫn: có thể ghép token từ nguồn sai vào câu trả lời có vẻ nhất quán.

</details>

**6. Với $p_\eta(z|x) = (0.5, 0.3, 0.2)$ và $p_\theta(y|x,z) = (0.6, 0.1, 0.0)$, tính $p_{\text{RAG-Seq}}(y|x)$ và gradient REALM theo điểm của từng tài liệu.**

<details markdown="1"><summary>Gợi ý đáp án</summary>

$p(y|x) = 0.30 + 0.03 + 0 = 0.33$. Posterior $= (0.30, 0.03, 0)/0.33 = (0.909, 0.091, 0)$. Gradient $= (0.909 - 0.5,\ 0.091 - 0.3,\ 0 - 0.2) = (+0.409, -0.209, -0.2)$: tăng điểm tài liệu 1, giảm hai tài liệu còn lại.

</details>

**7. Vì sao FiD có thể dùng 100 đoạn văn trong khi nối thẳng 100 đoạn vào encoder thì không?**

<details markdown="1"><summary>Gợi ý đáp án</summary>

Self-attention bậc hai theo độ dài. Nối $k$ đoạn dài $\ell$: $O(k^2\ell^2)$. FiD encode riêng: $O(k\ell^2)$ — với $k = 100$, rẻ hơn 100 lần. Decoder cross-attend lên toàn bộ trạng thái ẩn (tuyến tính theo $k\ell$ mỗi bước), nên tổng hợp được nhiều nguồn.

</details>

**8. RETRO và Atlas gợi ý điều gì về việc phân bổ ngân sách giữa "model to hơn" và "retrieval tốt hơn"?**

<details markdown="1"><summary>Gợi ý đáp án</summary>

RETRO: kho truy xuất khổng lồ giúp model nhỏ đạt hiệu năng mô hình hóa ngôn ngữ ngang model lớn hơn ~25 lần. Atlas: 11B tham số với retrieval vượt model 540B trên QA few-shot. Với tác vụ nặng tri thức, đầu tư vào retrieval (dữ liệu, index, rerank) thường hiệu quả hơn tăng kích thước model.

</details>

**9. Kể tên bảy điểm hỏng của Barnett et al. và chỉ ra điểm hỏng cho tình huống: "email tiếng Nhật hỏi về tính năng xuất CSV; bài hướng dẫn tiếng Anh có trong kho nhưng không được truy xuất".**

<details markdown="1"><summary>Gợi ý đáp án</summary>

Missing content, missed top-ranked documents, not in context, not extracted, wrong format, incorrect specificity, incomplete. Tình huống là FP2 (missed top-ranked): nội dung có trong kho nhưng không lọt top-$k$, do khoảng cách xuyên ngôn ngữ. Xử lý: embedding đa ngữ, hybrid, viết lại/dịch query.

</details>

**10. Phân biệt Naive, Advanced, Modular RAG. Bạn sẽ bắt đầu ở mức nào cho Zendesk và vì sao?**

<details markdown="1"><summary>Gợi ý đáp án</summary>

Naive: 3 bước tuyến tính. Advanced: thêm tối ưu trước truy xuất (chunking, metadata, viết lại query) và sau truy xuất (rerank, nén). Modular: đồ thị module có routing, tool, vòng lặp. Bắt đầu ở Advanced có đo lường vì đa số ticket là FAQ/hướng dẫn; chỉ đưa thành phần Modular (tool tra tài khoản, truy xuất lặp) vào các luồng mà số liệu cho thấy cần.

</details>

**11. Vì sao bài toán Zendesk không phải "chỉ RAG"? Viết dạng tổng quát của quyết định escalate.**

<details markdown="1"><summary>Gợi ý đáp án</summary>

Cần (A) phân loại trước (ngôn ngữ, intent, độ nhạy, muốn gặp người, injection), (B) RAG để sinh có căn cứ, (C) quyết định hành động. RAG thuần luôn sinh câu trả lời, không tôn trọng yêu cầu gặp người, không có quy tắc cho intent nhạy cảm, và không có ngưỡng để triển khai theo giai đoạn. Quyết định: $a^* = \arg\min_a\sum_s C(a,s)P(s|x,\hat y)$ với $a \in \{\text{gửi}, \text{draft}, \text{escalate}\}$.

</details>

---

## Bài tập thực hành

**Bài 1 — Đo hallucination closed-book vs RAG (GPU 6 GB hoặc API).**
Tự viết một "Help Center" giả lập gồm 15 bài ngắn (tiếng Việt và tiếng Anh) cho một sản phẩm SaaS hư cấu, có chính sách hoàn tiền, giá gói, giới hạn user, SLA — với con số *cố ý khác* với thông lệ (ví dụ hoàn tiền 9 ngày). Viết 30 câu hỏi kèm đáp án, trong đó 5 câu không có đáp án trong kho. Chạy một model nhỏ (Qwen2.5-1.5B/3B-Instruct qua vLLM hoặc Ollama) ở hai chế độ: (a) closed-book, (b) đưa đúng bài liên quan vào context. Gán nhãn tay từng câu trả lời: đúng / intrinsic / extrinsic / abstain đúng. Báo cáo bảng tỉ lệ và nhận xét riêng về 5 câu "không có đáp án".

**Bài 2 — Thí nghiệm lost in the middle thu nhỏ (GPU 6 GB).**
Dùng kho ở Bài 1. Với 10 câu hỏi, tạo context gồm 1 bài liên quan + 9 bài gây nhiễu (chọn các bài cùng chủ đề càng tốt), đặt bài liên quan ở vị trí 1, 5, 10. Đo độ chính xác theo vị trí (3 vị trí × 10 câu × 3 lần lặp với $T = 0$ và thứ tự nhiễu khác nhau). Vẽ đường theo vị trí. Có hình chữ U không? Model nhỏ với context ngắn có thể cho kết quả khác paper — hãy ghi rõ điều kiện, không suy rộng.

**Bài 3 — Tính tay và code RAG-Sequence / RAG-Token / posterior REALM (CPU).**
Viết hàm `rag_sequence(p_z, p_tok)` và `rag_token(p_z, p_tok)` với `p_z` có dạng `(k,)` và `p_tok` có dạng `(k, m)` (xác suất từng token đích dưới từng tài liệu), cùng `realm_grad(p_z, p_tok)` trả về $p(z|x,y) - p(z|x)$. Kiểm tra với ví dụ ở mục 5.3 (0,468; 0,395; (+0,337, −0,206, −0,131)). Sau đó tìm một ví dụ trong đó RAG-Token > RAG-Sequence và giải thích bằng lời.

**Bài 4 — Bảng ước lượng chi phí và ma trận quyết định cho đội của bạn (không cần GPU).**
Lập bảng tính (Python hoặc spreadsheet) với tham số đầu vào: số ticket/ngày, số lượt, token/lượt cho từng chiến lược, giá input/output, tỉ lệ cache hit. Tính chi phí/ngày và chi phí/ticket cho: RAG, long-context toàn Help Center, long-context + caching. Viết 1 trang biện luận chiến lược cho từng nguồn tri thức (theo bảng mục 7.3) với số liệu của bạn — dùng lại trong Capstone (Module 12).

---

## Tài liệu tham khảo

**Hallucination**

- Maynez, J., Narayan, S., Bohnet, B., McDonald, R. (2020). *On Faithfulness and Factuality in Abstractive Summarization*. ACL 2020. arXiv:2005.00661.
- Ji, Z. et al. (2023). *Survey of Hallucination in Natural Language Generation*. ACM Computing Surveys. arXiv:2202.03629.
- Huang, L. et al. (2023). *A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions*. arXiv:2311.05232.
- Kandpal, N., Deng, H., Roberts, A., Wallace, E., Raffel, C. (2023). *Large Language Models Struggle to Learn Long-Tail Knowledge*. ICML 2023. arXiv:2211.08411.
- Xu, Z., Jain, S., Kankanhalli, M. (2024). *Hallucination is Inevitable: An Innate Limitation of Large Language Models*. arXiv:2401.11817.
- Gekhman, Z. et al. (2024). *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?*. EMNLP 2024. arXiv:2405.05904.
- Kalai, A. T., Nachum, O., Vempala, S. S., Zhang, E. (2025). *Why Language Models Hallucinate*. arXiv:2509.04664.

**Context dài**

- Liu, N. F. et al. (2024). *Lost in the Middle: How Language Models Use Long Contexts*. TACL 12. arXiv:2307.03172.
- Hsieh, C.-P. et al. (2024). *RULER: What's the Real Context Size of Your Long-Context Language Models?*. COLM 2024. arXiv:2404.06654.
- Hong, K., Troynikov, A., Huber, J. (2025). *Context Rot: How Increasing Input Tokens Impacts LLM Performance*. Báo cáo kỹ thuật Chroma, 07/2025. https://www.trychroma.com/research/context-rot

**So sánh chiến lược**

- Ovadia, O., Brief, M., Mishaeli, M., Elisha, O. (2024). *Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs*. EMNLP 2024. arXiv:2312.05934.
- Li, Z. et al. (2024). *Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach*. EMNLP 2024 (Industry). arXiv:2407.16833.
- Chan, B. J. et al. (2024). *Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks*. arXiv:2412.15605.

**Hình thức hóa RAG**

- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020. arXiv:2005.11401.
- Guu, K., Lee, K., Tung, Z., Pasupat, P., Chang, M.-W. (2020). *REALM: Retrieval-Augmented Language Model Pre-Training*. ICML 2020. arXiv:2002.08909.
- Izacard, G., Grave, E. (2021). *Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering* (Fusion-in-Decoder). EACL 2021. arXiv:2007.01282.
- Borgeaud, S. et al. (2022). *Improving language models by retrieving from trillions of tokens* (RETRO). ICML 2022. arXiv:2112.04426.
- Izacard, G. et al. (2023). *Atlas: Few-shot Learning with Retrieval Augmented Language Models*. JMLR 24. arXiv:2208.03299.

**Pipeline, điểm hỏng, phân loại**

- Barnett, S., Kurniawan, S., Thudumu, S., Brannelly, Z., Abdelrazek, M. (2024). *Seven Failure Points When Engineering a Retrieval Augmented Generation System*. CAIN 2024. arXiv:2401.05856.
- Gao, Y. et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey*. arXiv:2312.10997.
- Gao, Y., Xiong, Y., Wang, M., Wang, H. (2024). *Modular RAG: Transforming RAG Systems into LEGO-like Reconfigurable Frameworks*. arXiv:2407.21059.

**Tài liệu chính thức**

- Anthropic — Prompt caching (giá ghi/đọc cache, TTL; tính đến 10/2026): https://platform.claude.com/docs/en/build-with-claude/prompt-caching
