# Implementation Plan: Porting InstantJPDict to Steam Deck with Decky Shim

This document outlines the step-by-step implementation plan for porting InstantJPDict to Steam Deck using a Rust-based daemon, Gamescope integration, and a Decky Loader plugin.

> **Note on UI Rendering**: A transparent overlay is not recommended as gamescope does not handle transparency correctly. Thus, the fullscreen interface or overlay should be designed with this constraint in mind (e.g., solid windows, capturing the background image and rendering it under ImGui elements, or drawing solid bounding boxes over a frozen/paused copy of the screen capture, or displaying cards on a solid side panel).

---

## Phase 1: The OCR Core & Logit Extraction (No UI/No Screen Capture)

### Goal
Isolate and test your 2-stage custom ONNX workflow inside a pure Rust command-line tool using a mock screenshot file.

### 🛠️ Implementation Steps
1. Create a workspace with a binary target (`accessibility_daemon`).
2. Add the `ort` crate to your `Cargo.toml`. Configure it for local dynamic or static linking depending on target footprint.
3. Port your Kotlin bounding box logic into Rust using the `image` or `ndarray` crate.
4. Run your first model to extract the spatial coordinates. Crop matching bounding sub-images directly from the native memory buffers.
5. Feed those sub-images into your second ONNX text-character recognition model. Ensure the `ort` session outputs the raw probability array directly. Write your specialized logit evaluation routines on top.

### 🧪 Test Plan
* **Test 1.1 (Inference Fidelity)**: Feed a reference static PNG text image via the command line (`cargo run -- ./test.png`). Verify that the extracted array shapes match your Android implementation exactly down to the raw floating-point logit precision.
* **Test 1.2 (Leak Check)**: Run the pipeline continuously inside a simple loop over 50 iterations. Use `htop` to confirm that memory consumption remains uniform.

---

## Phase 2: Input Capture & Gamescope PipeWire Integration

### Goal
Programmatically pull frames out of Gamescope's underlying streaming server using native interfaces.

### 🛠️ Implementation Steps
1. Add the `pipewire` or specialized frame capture crates to handle desktop portal handshakes.
2. Use Gamescope's system protocols to discover the active compositor rendering node.
3. Open a real-time data frame buffer pipe hook. Instead of a video stream, capture a single video frame when the hook triggers. Convert the raw multi-channel buffer layout (typically BGRA or RGBA) into an uncompressed memory array.

### 🧪 Test Plan
* **Test 2.1 (Desktop Environment Frame Grab)**: Execute your test binary on Linux. Verify it dumps a clean, pixel-accurate copy of your active desktop workspace straight to disk (`/tmp/capture.png`) without errors.
* **Test 2.2 (Steam Deck Game Mode Test)**: Push the binary to the Deck using SSH. Run a game, execute the command via terminal, and confirm that the active, hardware-accelerated viewport game screen drops properly without causing lag.

---

## Phase 3: Hardware Hooking & Input Hijack (libevdev)

### Goal
Read global hardware inputs from the Deck's physical game controller handles.

### 🛠️ Implementation Steps
1. Add the `evdev` crate to read system-wide `/dev/input/event*` streams natively.
2. Initialize an input device scraping scanner that listens to button state mappings globally.
3. Establish your trigger combo logic (e.g., holding L3 + R3 for 400ms). When detected, immediately fire your Phase 2 screenshot trigger to feed into your Phase 1 OCR engine.

### 🧪 Test Plan
* **Test 3.1 (Chord Detection)**: Run your background binary via SSH. Tap buttons on your console inside a running game. Confirm that the terminal logs the exact trigger event immediately when your custom chord threshold is met.

---

## Phase 4: Fullscreen ImGui Overlay Interface

### Goal
Draw interactive nested bounding boxes and scrollable window elements over the live screen using minimal assets.

### 🛠️ Implementation Steps
1. Add `sdl3` and `imgui-rs` bindings to handle lightweight graphics rendering loops.
2. Configure an opaque/solid background always-on-top window view via SDL window settings.
3. **The Layer Trick**: When the overlay is inactive, your window is invisible (or not spawned/fully hidden). When the Phase 3 trigger fires, instantly activate the SDL window surface.
4. Render the screenshot as a background layer, then use ImGui's drawing canvas primitives (`ImDrawList`) to outline bounding boxes over identified words. Use standard nested components like `ImGui::BeginChild` to implement your scrollable dictionary cards cleanly.

### 🧪 Test Plan
* **Test 4.1 (Focus and Interaction)**: Ensure that clicking or navigating with the D-pad selects boxes flawlessly, pops open your scrollable text drawer, and does not leak inputs down to the game engine underneath.
* **Test 4.2 (Dismissal Velocity)**: Test hitting a close shortcut or clicking outside to completely release window control back to Gamescope.

---

## Phase 5: Packaged Decky Bundle Deployment

### Goal
Merge your complete Rust pipeline binary and configuration scripts cleanly into a single-click Decky Loader store plugin footprint.

### 🛠️ Implementation Steps
1. Set up a standard Decky template repository structure containing your React UI frontend asset configurations.
2. Drop your cross-compiled production release build Rust binary alongside your ONNX model weights inside the plugin structure (`bin/`, `models/`).
3. Set up the plugin's `backend/main.py` daemon lifecycle handlers. Hook into the `_main` entry event to initialize your Rust child process loop in the background, and map `_unload` routines to terminate it safely on exit.

### 🧪 Test Plan
* **Test 5.1 (Production Install Verification)**: Copy your complete unpacked plugin structure straight into the target filesystem paths over at `/home/deck/homebrew/plugins/`.
* **Test 5.2 (Functional Lifecycle Audit)**: Restart your Deck. Verify that the plugin activates immediately, intercept keys inside any title, presents your fully-scrolled nested translations layout, and wipes itself completely from system memory upon plugin reload or removal.
