# De xuat cai thien SimLingo cho de tai mixed traffic

Ngay doc: 2026-04-26

Nguon chinh da doc:
- `papers/MustRead_Proposal.pdf`
- `papers/SimLingo.pdf`
- `papers/Bench2Drive.pdf`
- `papers/Think2Drive.pdf`
- Cac references trong `Bench2Drive_References`, `SimLingo_References`, `Think2Drive_References` duoc skim theo abstract/intro/method, tap trung vao VLM driving, world model, imitation learning, closed-loop benchmark, causal confusion, hidden bias, uncertainty-aware planning, va traffic physics.

## 1. Ket luan ngan

Co the cai thien SimLingo cho de tai nay. Huong manh nhat khong phai la "doi backbone VLM lon hon", ma la bien SimLingo thanh mot model co kha nang thich nghi voi unstructured mixed traffic bang du lieu counterfactual + rang buoc vat ly/an toan.

De tai nen di theo huong:

**Mixed-Traffic Interaction Dreaming (MTID)**  
Mo rong Action Dreaming cua SimLingo tu "ego tuong tuong nhieu hanh dong khac nhau trong cung mot visual context" sang "ego va cac tac nhan xung quanh cung co nhieu tuong lai vat ly kha thi", dac biet cho xe may, xe dap, nguoi di bo, jaywalking, chen lan, duong khong vach lane ro rang.

Day la huong co dong gop ro:
1. Phu hop proposal: domain gap tu structured traffic sang unstructured mixed traffic.
2. Dua tren SimLingo nen kha thi trong repo hien tai.
3. Co thanh phan moi ve toan/vat ly: kinematic bicycle model, IDM/Gipps, Social Force/ORCA, TTC/RSS/Control Barrier Function.
4. De danh gia duoc bang closed-loop CARLA/Bench2Drive style metrics va per-skill mixed-traffic metrics.

## 2. Tom tat nhung paper chinh

### SimLingo

SimLingo la Vision-Language-Action model dung camera-only, dua tren InternVL2. Model thong nhat 3 viec:
- Closed-loop driving.
- Vision-language understanding: VQA, commentary.
- Language-action alignment: Action Dreaming.

Diem quan trong cua SimLingo:
- Input: front camera, speed, target points hoac language command.
- Output: tach thanh 2 loai waypoint:
  - temporal speed waypoints de dieu khien toc do.
  - geometric path waypoints de dieu khien lateral/path.
- Action Dreaming tao nhieu instruction-action pair cho cung mot visual context bang offline non-reactive simulation, kinematic bicycle model, PID controller, va collision check.
- Action Dreaming giup model "nghe instruction" thay vi chi nhin visual cues.

Han che lien quan truc tiep de tai:
- SimLingo van chu yeu nam trong CARLA structured scenarios.
- Action Dreaming hien tai chu yeu thay doi ego trajectory, trong khi cac actor khac duoc replay/non-reactive.
- Chua tap trung vao xe may/xe dap/nguoi di bo di hon loan, khong theo lane, khong nhung duong.
- CoT/commentary chi cai thien nhe va chua duoc khai thac thanh reasoning supervision co cau truc.
- Sim-only, chua co domain adaptation ro cho mixed traffic.

### Bench2Drive

Bench2Drive nhan manh:
- Open-loop L2 error khong du de danh gia driving.
- Closed-loop evaluation moi phan anh duoc loi distribution shift, causal confusion, va kha nang tuong tac.
- Benchmark chia 220 short routes, 44 scenarios, 5 ability groups: Merging, Overtaking, Emergency Brake, Give Way, Traffic Sign.
- Per-skill evaluation huu ich hon mot driving score trung binh duy nhat.

Bai hoc cho de tai:
- Khong nen chi bao cao L2/ADE/FDE.
- Can tao benchmark mixed-traffic theo skill, vi domain gap khong dong deu tren moi tinh huong.

### Think2Drive

Think2Drive la model-based RL planner voi latent world model:
- Dung privileged BEV/HD-map/actor states, khong tap trung perception.
- Hoc world model dang RSSM/DreamerV3 de rollout trong latent space.
- Planner hoc bang "thinking" trong world model thay vi rollout cham trong CARLA.
- Cac brick quan trong: automated scenario generation, reset planner de tranh local optimum, termination-priority replay, steering cost.

Bai hoc cho de tai:
- Scenario-dense generation la rat quan trong cho long-tail driving.
- World model co the dung nhu teacher/planner/scorer cho SimLingo, khong nhat thiet phai thay SimLingo.
- Can reward/risk signal day du hon imitation loss.

## 3. Chan doan nhanh repo hien tai

Co hai viec nen sua/kiem tra truoc khi nghien cuu thuat toan lon:

1. **Bucket training dang co dau hieu sai phan phoi.**  
   Log `logs/finetune_simlingo_v2_all_20260426_114630.log` cho thay nhieu bucket quan trong load `0 images`, vi du `vehicle_side`, `leading_object_vehicle`, `changed_route`, `parkinglane`.  
   `database/bucketsv2_simlingo/buckets_paths.pkl` dang tro den `database/simlingo_v2_2026_04_21/...`, trong khi training log/config dang dung `database/simlingo_v2_all`. Code trong `simlingo_training/dataloader/dataset_base.py` chi hard-code replace tu `database/simlingo_v2_2025_01_10`, nen bucket path khong map sang dataset hien tai.  
   Tac dong: model co the dang train chu yeu tren bucket `all` va mat phan long-tail/hazard sampling, trai voi recipe cua SimLingo.

2. **Eval hien tai chua du ket luan.**  
   `eval_results.json` chi co 1 route `ParkingCutIn_1`, DS = 25.94, RC = 74.46, bi `Agent got blocked`, co layout collision va outside-route-lane. Day la tin hieu debug, khong phai ket qua benchmark hoan chinh. Can chay du 220 routes Bench2Drive hoac mot mixed-traffic eval suite rieng.

Thu tu nen lam truoc:
- Regenerate bucket file cho dung dataset hien tai, hoac update config ve dung dataset ma bucket file tro toi.
- Kiem tra lai so sample moi bucket.
- Chay baseline closed-loop du nho nhat: 20-50 routes truoc, roi 220 routes.

## 4. De xuat nghien cuu chinh: Mixed-Traffic Interaction Dreaming

### 4.1 Y tuong

Action Dreaming cua SimLingo tao cac ego future khac nhau cho cung mot frame. MTID mo rong thanh:

> Voi cung mot visual context, tao nhieu tuong lai vat ly kha thi cho ca ego va cac actor xung quanh, roi train SimLingo de chon hanh dong an toan, mem, va phu hop instruction trong mixed traffic.

Thay vi chi hoi:
- "Neu lenh la lane change left thi ego trajectory ra sao?"

MTID hoi:
- "Neu xe may ben trai chen vao khoang trong 1.2s nua thi ego nen giam toc/giu khoang cach/lech nhe the nao?"
- "Neu nguoi di bo bat dau jaywalk tu via he, ego co nen tiep tuc, giam toc, hay dung?"
- "Neu duong khong co vach lane, ego nen giu corridor nao de tranh xe may nguoc chieu?"

### 4.2 Thanh phan vat ly/toan hoc

Dung cac model don gian nhung co nghia:

**Ego dynamics: kinematic bicycle**

State:
`x = [px, py, yaw, v]`

Control:
`u = [steer, throttle/brake]`

Update gan voi code hien tai trong `team_code/agent_simlingo.py`:
`x_{t+1} = f_bicycle(x_t, u_t)`

**Vehicle/car following: IDM hoac Gipps**

Gia toc xe sau:

`a = a_max * (1 - (v / v0)^delta - (s_star(v, delta_v) / s)^2)`

Trong do:
`s_star = s0 + v*T + v*delta_v / (2*sqrt(a_max*b))`

IDM phu hop voi cac tinh huong dong xe, bottleneck, congestion, merge.

**Pedestrian/bike/motorbike interaction: Social Force / ORCA-lite**

Moi actor co desired velocity va luc tranh va cham:

`F_i = F_goal + sum_j F_repulse(i,j) + F_boundary`

Voi xe may, co the them lateral filtering behavior:
- Cho phep di giua 2 lane/corridor.
- Toc do muc tieu lon hon pedestrian, nho hon/gan car.
- Khoang cach an toan bat doi xung: xe may can it lateral gap hon xe hoi, nhung nguy hiem hon pedestrian do toc do cao.

**Risk function**

Voi moi actor `i`, tinh:
- distance `d_i`
- relative speed `dv_i`
- time-to-collision `TTC_i`
- required deceleration `DRAC_i`
- heading conflict `cos(theta_rel)`

Risk:

`R_i = exp(-d_i / sigma_d) * exp(-TTC_i / sigma_t) * g(class_i, heading_i)`

Tong risk:

`R_total = sum_i w_class(i) * R_i`

**Safety barrier**

Dung Control Barrier Function dang don gian:

`h_i(x) = d_i(x)^2 - d_safe_i^2`

An toan neu:

`h_i(x) >= 0`

Phat trong loss neu duong di du doan vi pham:

`L_barrier = sum_t sum_i ReLU(d_safe_i^2 - d_i(t)^2)`

**Traffic pressure / potential field**

Dinh nghia mot truong chi phi:

`U(x) = w_route U_route(x) + w_obs sum_i phi_i(x) + w_lane U_lane(x) + w_comfort U_comfort(x)`

Trong mixed traffic, `U_lane` khong nen qua cung, vi duong co the khong ro lane. Thay vao do dung **drivable corridor**:

`U_corridor = distance_to_corridor_center^2`, voi corridor co the sinh tu route + road boundary + obstacle density.

### 4.3 Cach sinh MTID data

Moi sample trong dataset co:
- RGB front image.
- Ego state/speed.
- Route/target points.
- Actor states neu co trong measurement/boxes.
- Scenario tag: motorcycle cut-in, jaywalking, lane-less, wrong-way bike, dense mixed traffic, parked obstacle + motorbike flow, etc.

Voi moi sample:
1. Lay actor quanh ego trong ban kinh 30m.
2. Sinh K tuong lai cho actor:
   - normal following
   - aggressive cut-in
   - sudden jaywalk
   - slow bicycle wobble
   - motorcycle lane filtering
   - wrong-way encroachment
3. Sinh M ung vien ego action:
   - keep speed
   - slow down
   - brake
   - nudge left/right
   - yield
   - follow corridor
4. Rollout ego bang kinematic bicycle, actor bang IDM/Social Force/ORCA-lite.
5. Cham diem bang:
   - collision/TTC/RSS risk
   - route progress
   - comfort
   - off-road/corridor violation
6. Chon trajectory safe/best lam label, va tao instruction/commentary:
   - "Slow down and keep a wider gap because a motorcycle may cut in from the left."
   - "Do not move right; a pedestrian is likely to cross from the sidewalk."
   - "Follow the open corridor instead of the faded lane marking."
7. Tao negative/unsafe instructions giong SimLingo safety flag:
   - "Accelerate through the gap between the motorcycle and pedestrian."
   - Answer safety: "This is unsafe because TTC is below threshold..."

### 4.4 Loss de train

Giu loss goc cua SimLingo:

`L_wp = SmoothL1(pred_speed_wps, gt_speed_wps) + SmoothL1(pred_path, gt_path)`

Them:

`L_safe = sum_t sum_i ReLU(d_safe_i^2 - d_i(t)^2)`

`L_ttc = sum_t sum_i ReLU(TTC_min - TTC_i(t))`

`L_domain = CE(domain_head(features), domain_label)`

Neu dung domain adversarial:

`L = L_wp + lambda_lang L_lang + lambda_safe L_safe + lambda_ttc L_ttc - lambda_adv L_domain`

Muc tieu:
- Features bot phu thuoc vao "structured CARLA only".
- Waypoints tuan thu rang buoc an toan vat ly.
- Language/action alignment khong chi la instruction following ma la interaction-aware.

## 5. Cac huong cai thien theo muc do rui ro

### Quick wins

1. Sua bucket path / regenerate bucket cho dataset hien tai.
2. Tang ty le bucket cho long-tail: walker, vehicle_side, changed_route, start_from_stop, parkinglane, motorcycle/bicycle neu co.
3. Them per-scenario eval report thay vi chi DS trung binh.
4. So sanh 3 che do inference:
   - target_point
   - command
   - target_point_command
   - CoT on/off
5. Kiem tra controller PID voi dataset moi, vi SimLingo paper cho thay tuned controller co the tang DS rat lon.

### Medium-risk improvements

1. Mixed-traffic commentary/VQA templates:
   - "motorcycle filtering from left/right"
   - "jaywalker near curb"
   - "vehicle/bike does not yield"
   - "faded lane/no lane marking"
2. Mixed-traffic Dreamer modes:
   - yield-to-motorbike
   - avoid-jaywalker
   - follow-corridor
   - cautious-gap-acceptance
   - nudge-around-parked-obstacle-with-oncoming-bike
3. Safety auxiliary loss tren predicted waypoints.
4. Domain tags + weighted sampler.

### High-risk / paper-worthy

1. MTID nhu tren.
2. Latent world model scorer:
   - Train mot world model nho tren BEV actor states/masks.
   - Dung no de cham diem N candidate trajectories cua SimLingo.
   - Khong can thay VLM; chi them plan selection / safety reranking.
3. Preference-based reward:
   - Tao pairwise preferences giua 2 trajectory: safe/smooth/progress.
   - Train reward model de rerank output SimLingo.

## 6. Benchmark de xuat cho de tai

Ngoai Bench2Drive 5 abilities, them mixed-traffic abilities:

1. Motorcycle Cut-in
2. Motorcycle Lane Filtering
3. Jaywalking Pedestrian
4. Bicycle Wobble / Slow Bicycle Overtake
5. Lane-less Corridor Following
6. Wrong-way Two-wheeler
7. Dense Unsignalized Junction
8. Parked Obstacle + Oncoming Flow

Metric:
- DS, RC, IS theo CARLA/Bench2Drive.
- SR theo skill.
- Collision per km.
- TTC violation rate.
- Comfort: jerk, lateral acceleration.
- Domain gap:

`Gap = Score_structured - Score_mixed`

Muc tieu proposal co the viet ro:

`Improvement = (Score_MTID - Score_baseline) / Score_baseline`

Ky vong hop ly: +10-15% route completion / success rate trong mixed traffic, giong proposal da neu.

## 7. De xuat dong gop khoa hoc

Ten co the dung:

**SimLingo-MT: Mixed-Traffic Interaction Dreaming for Vision-Language-Action Autonomous Driving**

Dong gop:
1. Mot mixed-traffic CARLA scenario suite cho developing-country traffic.
2. Mot data generation method: Mixed-Traffic Interaction Dreaming.
3. Mot safety/physics-aware training objective cho SimLingo.
4. Mot benchmark phan tich domain gap structured -> unstructured mixed traffic.

So voi Action Dreaming:
- Action Dreaming: ego counterfactual, actor replay/non-reactive.
- MTID: ego + actor counterfactual, interaction-aware, co risk/barrier physics.

So voi Think2Drive:
- Think2Drive: privileged latent world model/RL planner.
- MTID: khong can full RL teacher, chi dung physics rollout offline de tao supervision va safety loss cho VLA model.

So voi Bench2Drive:
- Bench2Drive: multi-ability closed-loop benchmark.
- MTID benchmark: multi-ability nhung tap trung mixed/unstructured traffic.

## 8. Lo trinh thuc hien de xuat

### Giai doan 1: Reproduce baseline

- Fix bucket/data path.
- Chay SimLingo baseline tren mot subset Bench2Drive.
- Ghi DS/SR/RC/IS + per-scenario errors.
- Chon 5-8 scenarios lien quan mixed traffic.

### Giai doan 2: Mixed-traffic scenario generation

- Spawn motorcycles/bicycles/pedestrians trong CARLA Traffic Manager.
- Tao route ngắn 100-200m, moi route mot hazard chinh.
- Luu actor states, boxes, measurements, RGB.
- Tao scenario tags.

### Giai doan 3: MTID label generation

- Implement offline rollout:
  - ego kinematic bicycle.
  - car IDM.
  - pedestrian/bike/motorbike Social Force/ORCA-lite.
  - risk/TTC/barrier scoring.
- Tao dreamer json cung format gan voi SimLingo Dreamer.
- Tao templates instruction/commentary.

### Giai doan 4: Fine-tune

- Train baseline SimLingo tren structured + mixed traffic.
- Train SimLingo + MTID.
- Ablation:
  - no MTID
  - MTID without actor counterfactual
  - MTID without safety loss
  - MTID full

### Giai doan 5: Evaluation

- Closed-loop mixed suite.
- Bench2Drive subset de kiem tra khong lam hong kha nang structured driving.
- Bao cao domain gap va improvement.

## 9. Cau tra loi truc tiep cho cau hoi trong MustRead

**Co the cai thien hieu suat SimLingo cho de tai nay khong?**  
Co. Truoc mat co quick win ve bucket/data/controller/eval. Ve nghien cuu, SimLingo chua duoc toi uu cho unstructured mixed traffic, nen con dat de cai thien bang data generation, domain adaptation, va safety-aware loss.

**Co the cai thien thuat toan khong?**  
Co. Huong thuat toan nen la MTID: them actor counterfactual + risk physics + barrier loss, thay vi chi imitation learning waypoint.

**Co the nghi ra cai gi hay nhu Action Dreaming bang toan/vat ly thuan tuy khong?**  
Co. MTID la ung vien tot: no mo rong Action Dreaming bang kinematic bicycle, IDM, Social Force/ORCA-lite, TTC/RSS/CBF, va potential-field risk. No du "mathematical/physical" de thanh dong gop rieng, nhung van gan truc tiep voi SimLingo va dataset hien co.

## 10. Viec nen lam tiep ngay trong repo

1. Regenerate `database/bucketsv2_simlingo/buckets_paths.pkl` cho `database/simlingo_v2_all`, hoac doi config training ve dataset ma bucket file dang tro toi.
2. Tao script thong ke bucket coverage truoc moi lan train.
3. Chay eval subset 20 routes va parse per-scenario failure.
4. Tao prototype `dataset_generation/dreamer_data/mixed_traffic_dreamer_generator.py` dua tren generator Dreamer hien co.
5. Them mixed-traffic templates vao `data/augmented_templates/dreamer.json` hoac file rieng `mixed_traffic_dreamer.json`.