<p align="center">
  <img src="docs/assets/readme/v3/hero.gif" alt="MaleCNS / OBS Fly" width="100%">
</p>

<p align="center">
  <strong>A tiny OBS fly with a body, a life, a behavior system — and eventually a brain.</strong><br>
  Character · physical OBS habitat · autonomous life · stream events · MaleCNS-inspired reflex experiments
</p>

<p align="center">
  <a href="#-meet-the-fly">Character</a> ·
  <a href="#-obs-is-the-habitat">OBS Habitat</a> ·
  <a href="#-the-flys-daily-life">Daily Life</a> ·
  <a href="#-life-dashboard">Life Dashboard</a> ·
  <a href="#-behavior-example-flow">Behavior Flow</a> ·
  <a href="#-brain--control-stack">Brain</a> ·
  <a href="#-stream-events">Events</a> ·
  <a href="#-current-development">Current Phase</a> ·
  <a href="#-full-roadmap">Roadmap</a>
</p>

---

## 🪰 What is MaleCNS / OBS Fly?

What if a tiny cartoon fly actually **lived inside your OBS scene**?

Not as a looping GIF.  
Not as a random alert animation.

The goal is a character that has:

- a physical body
- a physical world
- internal needs
- autonomous daily behavior
- stream-event reactions
- subscriber commands
- and, later, a MaleCNS-inspired steering/reflex layer underneath the high-level life system

The intended loop is:

**walk → fly → land → collide → sit → watch → eat → sleep → train → clean → react → continue living**

OBS is not only where the character is rendered. **The OBS scene itself becomes the habitat.**

> This is an experimental student project. It does not claim consciousness, a biologically validated complete fly brain, or a finished scientific simulation.

---

## ✨ Meet the Fly

<p align="center">
  <img src="docs/assets/readme/v3/character-showcase.png" alt="MaleCNS Blender fly character from multiple angles" width="100%">
</p>

The current fly is an original **low-poly cartoon character** built for small-screen readability inside a stream overlay.

The exported `fly_master.glb` currently contains:

| Character data | Current export |
| :--- | ---: |
| Animation clips | **7** |
| Animation channels per clip | **81** |
| Meshes | **63** |
| Nodes | **96** |
| Runtime format | **GLB / glTF 2.0** |

### Current animation clips

| Clip | Duration | Use |
| :--- | ---: | :--- |
| `FLY_IDLE` | 2.000 s | Ground idle loop |
| `FLY_WALK` | 1.000 s | Ground locomotion |
| `FLY_FLIGHT` | 0.333 s | Airborne loop |
| `FLY_TAKEOFF` | 0.708 s | Ground → flight |
| `FLY_LAND` | 0.792 s | Flight → ground |
| `FLY_TURN_LEFT` | 0.708 s | Left turn |
| `FLY_TURN_RIGHT` | 0.708 s | Right turn |

The GLB is already connected to the browser-facing OBS runtime.

---

## 🎬 OBS is the Habitat

<p align="center">
  <img src="docs/assets/readme/v3/obs-habitat.gif" alt="OBS support and collision examples" width="100%">
</p>

A core architecture rule is:

> **visual source ≠ physical collider ≠ support surface ≠ activity target**

That distinction matters because an OBS source can block the fly physically without being a valid place to stand, sit, place a chair or perform an activity.

### Current habitat rules

- `WORLD` is the habitat scene
- OBS sources participate in collision geometry
- Combo is fully ignored
- Mario has collision but no activity/support role
- Eventlist has collision but no activity/support role
- Webcam can remain a usable support surface
- Kick Chat geometry was corrected
- Kick Chat stays aligned during normal movement, manual movement and `WATCHING`
- Physical collider and activity/support semantics are being separated

A target sequence looks like:

**walk across chat → reach edge → fall → take off → land on webcam → place chair → sit → watch**

---

## 🏠 The Fly's Daily Life

<p align="center">
  <img src="docs/assets/readme/v3/daily-life.gif" alt="Watching, eating, sleeping, cardio, pushups and sweeping" width="100%">
</p>

The long-term goal is not to pick a random animation every few seconds.

The fly should have a small autonomous life:

- 🪑 `WATCHING`
- 🍗 `EATING / SNACKING`
- 😴 `RESTING / SLEEPING`
- 🏃 `CARDIO`
- 💪 `PUSHUPS`
- 💪 `SITUPS`
- 🧹 `SWEEPING`
- 🧼 `GROOMING`
- 🔎 `EXPLORING`

Behavior selection should depend on **internal state, cooldowns, physical conditions, costs and available surfaces**.

---

## 📊 Life Dashboard

<p align="center">
  <img src="docs/assets/readme/v3/life-dashboard.gif" alt="Animated concept for the Fly Life Dashboard" width="100%">
</p>

The planned life-state model includes:

| State | Intended role |
| :--- | :--- |
| `energy` | walking, flying and exercise cost energy; food/rest recover it |
| `hunger / satiety` | changes over time; eating restores satiety |
| `fatigue / sleep_pressure` | activity increases it; sleep/rest recover it |
| `strength / conditioning` | training can gradually affect physical capacity |
| `stress / arousal` | sensory events and reactions can modulate it |
| `boredom / activity_need` | inactivity can make exploration/activity more attractive |

The values shown in the animated dashboard are **illustrative UI values**, not measured runtime results.

### Character progression

Training is intended to be more than a visual gag.

Repeated exercise can eventually feed into `strength / conditioning`, while low energy and fatigue still limit what the character can do.

That gives the fly a tiny long-term development arc rather than resetting to exactly the same internal state forever.

---

## 🔀 Behavior Example Flow

<p align="center">
  <img src="docs/assets/readme/v3/behavior-flow.gif" alt="State driven behavior and event preemption example" width="100%">
</p>

Example:

**boredom rises → exploring becomes attractive → walk → edge → fall → takeoff → land on webcam → WATCHING**

Then a higher-priority event arrives:

**500 Kicks → priority 300 → cancel current daily behavior → cleanup → DANCE_PARTY → return to autonomous life**

This is why behavior lifecycle and cleanup are important. A preempted behavior should not leave a chair, treadmill, jail prop or stale animation state behind.

---

## 🧠 Brain / Control Stack

<p align="center">
  <img src="docs/assets/readme/v3/brain-control.gif" alt="Autonomous life and MaleCNS control stack" width="100%">
</p>

The design deliberately separates two kinds of control.

### Autonomous life — **WHAT is the fly doing?**

`Life State → Life Manager → Behavior Arbiter → Current Behavior`

The Life Manager can evaluate:

- current internal state
- cooldowns
- preconditions
- cost / benefit
- available surfaces
- current stream/event context

### MaleCNS / sensory reflex layer — **HOW does the body react while doing it?**

Later integration is intended to support:

- steering
- avoidance
- startle
- looming response
- locomotor variation
- arousal modulation

MaleCNS is therefore not planned as the owner of every high-level daily-life choice.

A behavior can continue while short reflexes temporarily modulate locomotion.

---

## 🎁 Stream Events

<p align="center">
  <img src="docs/assets/readme/v3/stream-events.gif" alt="Meal, Dance Party and Jail stream events" width="100%">
</p>

Planned Kicks routing:

- **100 Kicks → MEAL**
- **500 Kicks → DANCE_PARTY + existing meme/video**
- **1000 Kicks → JAIL for 15 minutes**

### 🚔 Jail concept

The jail event can include:

- Jail asset
- Enter animation/transition
- Idle
- Sit
- Sleep
- Tally marks
- Secret escape attempts
- Nail clipper
- Energy-based escape rate
- Escape energy cost
- `!kaçmaya çalışıyor`
- Hide tool
- Innocent pose
- Escape progress reset
- Sentence reset → 15 min

### 🟣 Subscriber commands

Planned command layer:

- Subscriber verification
- Whitelist
- Cooldown
- `!cardio`
- `!pushup`
- `!situp`
- Low-energy rejection
- `COMMAND_REJECTED_LOW_ENERGY`
- Rejection shown in the Brain / State UI

---

## 🚧 Current Development

<p align="center">
  <img src="docs/assets/readme/v3/development-progress.gif" alt="MaleCNS OBS Fly development progress" width="100%">
</p>

### **Current phase: Final Physics Calibration**

The current job is to stop using placeholder body dimensions and make physics match the **real exported GLB character**.

- Real GLB body collider
- Real landing point
- Physical foot contact point
- Walking footprint
- Standing footprint
- Flying collider
- Overlay-edge snag testing
- Left/right collision testing
- Landing on overlays
- Collision from below
- Walking off an overlay and falling
- `SCREEN_FLOOR`
- Ground ↔ flight transitions

---

# 🗺️ Full Roadmap

Status legend: **✅ implemented foundation** · **🔴 current** · **🟠 next** · **🟡 planned** · **⚪ later / integration**

<details open>
<summary><strong>✅ A. Karakter / Blender</strong></summary>

- Özgün low-poly cartoon sinek tasarlandı
- Rig oluşturuldu
- Core animasyonlar hazır:
  - `FLY_IDLE`
  - `FLY_WALK`
  - `FLY_FLIGHT`
  - `FLY_TAKEOFF`
  - `FLY_LAND`
  - `FLY_TURN_LEFT`
  - `FLY_TURN_RIGHT`
- `fly_master.blend` master dosya olarak kaydedildi
- `fly_master.glb` export edildi
- GLB OBS Browser Source'a bağlandı
- GLB animasyonları runtime'da çalışıyor
- Karakter scale düzeltildi
- Grounding / yere basma düzeltildi

</details>

<details open>
<summary><strong>✅ B. OBS Habitat — Temel Geometri</strong></summary>

- `WORLD` sahnesi habitat olarak kullanılıyor
- OBS source'ları collision sistemine giriyor
- kombo tamamen `IGNORE`
- Mario collision var ama activity/support yok
- Eventlist collision var ama activity/support yok
- Webcam kullanılabilir support olarak kalıyor
- Kick Chat geometry bug düzeltildi
- Kick Chat normal hareket sırasında doğru yerde
- Kick Chat manual movement sırasında doğru yerde
- Kick Chat `WATCHING` sırasında doğru yerde
- Physical collider ile activity/support kavramları ayrılmaya başladı

</details>

<details open>
<summary><strong>🔴 C. ŞİMDİKİ AŞAMA — Final Fizik Kalibrasyonu</strong></summary>

- Gerçek GLB karakter için final body collider
- Gerçek landing point
- Ayakların fiziksel temas noktası
- Walking footprint
- Standing footprint
- Flying collider
- Sineğin overlay kenarlarında takılma testi
- Sol/sağ collision testi
- Overlay üstüne landing testi
- Overlay altından çarpma testi
- Overlay üstünden yürüyüp düşme testi
- `SCREEN_FLOOR` davranışı testi
- Ground ↔ flight transition testi

Burada artık eski placeholder ölçülerini tamamen bırakıp gerçek karakter boyutuyla fizik yapacağız.

</details>

<details>
<summary><strong>🟠 D. WATCHING Finalizasyonu</strong></summary>

Fizik oturduktan sonra:

- Gerçek sandalye asset'i
- Sandalye boyutunu karaktere göre ayarla
- Sineğin sandalyeye doğru yürümesi/uçması
- Sandalyeye oturma noktası
- Webcam üstünde sandalye yerleşimi
- Chat üstünde sandalye yerleşimi
- Uygun olmayan küçük yüzeyleri reddet
- Aynı yere sürekli sandalye koymama
- Natural weighted placement
- `FLY_SIT_DOWN`
- `FLY_SIT_IDLE`
- `FLY_STAND_UP`
- `FLY_WATCHING_IDLE`

</details>

<details>
<summary><strong>🟡 E. Life State Sistemi</strong></summary>

- `energy`
- `hunger / satiety`
- `fatigue / sleep_pressure`
- `strength / conditioning`
- `stress / arousal`
- `boredom / activity_need`
- Zamanla state değişimi
- Walking enerji maliyeti
- Flying enerji maliyeti
- Exercise enerji/fatigue maliyeti
- Sleep recovery
- Food recovery
- Critical energy davranışları

</details>

<details>
<summary><strong>🟢 F. Life Manager / Autonomous Life</strong></summary>

- `RESTING`
- `EXPLORING`
- `WATCHING`
- `SLEEPING`
- `EATING / SNACKING`
- `GROOMING`
- `SWEEPING`
- `PUSHUPS`
- `SITUPS`
- Random değil, state-driven seçim
- Cooldown sistemi
- Preconditions
- Cost / benefit
- Behavior transition sistemi

</details>

<details>
<summary><strong>🔵 G. Behavior Arbiter</strong></summary>

Priority kesin:

```text
KICKS EVENTS        = 300
SUBSCRIBER COMMANDS = 200
DAILY LIFE          = 100
```

- `ENTER`
- `ACTIVE`
- `CANCEL_REQUESTED`
- `TRANSITION_OUT`
- `CLEANUP`
- `DONE`
- Graceful preemption
- Current behavior
- Pending behavior
- Context commands ayrı route

</details>

<details>
<summary><strong>🟣 H. Subscriber Commands</strong></summary>

- Subscriber verification
- Whitelist command sistemi
- Cooldown
- `!cardio`
- `!pushup`
- `!situp`
- Energy insufficient → reject
- `COMMAND_REJECTED_LOW_ENERGY`
- Brain/State UI'da rejection gösterimi

</details>

<details>
<summary><strong>🟪 I. Kicks Events</strong></summary>

- Kicks router
- Mevcut meme/video sistemini koru
- **100 Kicks → MEAL**
- **500 Kicks → DANCE_PARTY + mevcut video/meme**
- **1000 Kicks → JAIL 15 dakika**

### Jail

- Jail asset
- Jail enter
- Idle
- Sit
- Sleep
- Tally mark
- Secret escape
- Nail clipper
- Energy-based escape rate
- Escape energy cost
- `!kaçmaya çalışıyor`
- Tool hide
- Innocent pose
- Escape progress reset
- Sentence reset → 15 min

### Dance

- Dance floor
- Speakers
- Disco ball
- Dance animation set
- Event duration
- Return to previous life

</details>

<details>
<summary><strong>🟤 J. Props / Assets</strong></summary>

- Chair
- Popcorn
- Table
- Table cloth
- Food
- Blanket
- Lamp
- Broom
- Treadmill
- Jail
- Nail clipper
- Dance floor
- Speakers
- Disco ball

</details>

<details>
<summary><strong>⚫ K. MaleCNS Integration</strong></summary>

Bunu life sisteminden sonra bağlayacağız.

- High-level behavior ≠ MaleCNS
- MaleCNS steering
- Avoidance
- Startle
- Looming response
- Locomotor variation
- Arousal modulation
- Behavior devam ederken kısa reflex
- Energy/strength → actuator capacity modeli
- Controlled internal signals

</details>

<details>
<summary><strong>⚪ L. Vision</strong></summary>

- Local motion
- Motion left/right
- Looming
- Brightness flash
- Large scene change
- Temporal filtering
- Hysteresis
- Cooldown
- Semantic vision daha seyrek
- Vision doğrudan behavior seçmesin

</details>

<details>
<summary><strong>🧠 M. Brain / Life Dashboard</strong></summary>

- Current behavior
- Behavior phase
- Energy
- Hunger
- Fatigue
- Strength
- Stress/arousal
- Boredom
- Current animation
- Current support surface
- Collision state
- MaleCNS motor output
- Recent sensory event
- Subscriber/Kicks event log

</details>

<details>
<summary><strong>🧹 N. Final Refactor</strong></summary>

En son:

- `main.py`
- `config.py`
- `brain/`
- `body/`
- `behavior/`
- `daily_life/`
- `subscriber_commands/`
- `kicks_events/`
- `integrations/`
- `world/`
- `ui/`
- `assets/`
- Eski experiment'leri `experiments/` altına taşı
- Tek production entry point
- Config/constants temizliği
- Logging temizliği

</details>

---

## 🔐 Security / Privacy

This public repository should contain **configuration names and source code, never credential values**.

Keep API keys, OBS credentials, tokens, private screenshots and raw personal logs outside the repository.

The `.gitignore` already excludes common credential files, runtime logs, screen captures and generated scientific data. Review staged files before every public commit.

---

## 🔬 Where MaleCNS Fits

The repository also contains earlier experiments around:

- LIF neural simulation
- continuous and multi-input brain sessions
- visual retina / laterality
- local screen/visual encoding
- motor decoding / steering
- optional semantic perception
- browser / OBS embodiment

The current architecture direction is:

**stable body → stable world → life state → behavior lifecycle → MaleCNS reflex integration**

---

## 🗂️ Repository Map

```text
MaleCNS-Lab/
├── body/
├── world/
├── vision/
├── data/
├── docs/
├── assets/vendor/
├── fly_master.glb
├── lif_engine*.py
├── brain_session*.py
├── motor_decoder*.py
├── vision_bridge*.py
├── visual_retina_encoder*.py
└── obs_overlay_server*.py
```

The repository still contains versioned experiments. Final cleanup is intentionally postponed until the runtime architecture stabilizes.

---

## 🚀 Getting Started

Python 3.12 reflects the original development environment.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

This is still an experimental source release rather than a one-command end-user application.

See:

- [`data/README.md`](data/README.md)
- [`CONTRIBUTING.md`](CONTRIBUTING.md)

---

## 🤝 Help Shape It

Useful contribution areas include:

- physics / collision architecture
- behavior arbitration and graceful preemption
- OBS WebSocket geometry handling
- Three.js / GLB runtime behavior
- animation transitions
- life-state architecture
- connectome-to-motor experiments
- visual laterality
- profiling and reproducibility

---

<p align="center">
  <strong>Built by <a href="https://github.com/Yuntrix">Yuntrix</a></strong><br>
  CS student · PJATK, Warsaw · learning in public
</p>
