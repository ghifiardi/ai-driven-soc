# 🎥 Recording the Threat Hunting Simulation

Complete guide to record a professional terminal demo video, just like the AI SOC showcase.

---

## 📋 Prerequisites

### Required Tools

```bash
# 1. Install asciinema (terminal recorder)
brew install asciinema

# 2. Install agg (converts .cast to .gif)
brew install agg

# Or on Linux:
# sudo apt install asciinema
# cargo install --git https://github.com/asciinema/agg
```

---

## 🎬 Recording Process

### Step 1: Prepare Your Terminal

```bash
# 1. Navigate to the directory
cd /Users/raditio.ghifiardigmail.com/Downloads/ai-driven-soc.backup/

# 2. Clear terminal
clear

# 3. Set terminal size for best recording (optional but recommended)
# Resize your terminal window to: 120 columns × 40 rows
# You can check current size with:
echo "Columns: $(tput cols), Rows: $(tput lines)"
```

### Step 2: Start Recording

```bash
# Start asciinema recording
asciinema rec demo/recordings/threat_hunting_demo.cast

# You'll see:
# asciinema: recording asciicast to demo/recordings/threat_hunting_demo.cast
# asciinema: press <ctrl-d> or type "exit" when you're done
```

### Step 3: Run the Simulation

Now run the simulation and interact with it:

```bash
# Run the simulation
python3 demo_simulation.py

# Press ENTER at each prompt to progress through the phases
# The simulation will pause at:
# - After VALHALLA (Phase 1)
# - After ASGARD (Phase 2)
# - After THOR (Phase 3)
# - After TAA (Phase 4)
# - After CRA (Phase 5)
# - After CLA (Phase 6)
# - After Summary

# Total: 7 prompts - just press ENTER at each
```

### Step 4: Stop Recording

```bash
# When simulation is complete, press Ctrl+D to stop recording
# Or type: exit

# You'll see:
# asciinema: recording finished
# asciinema: asciicast saved to demo/recordings/threat_hunting_demo.cast
```

---

## 🎨 Convert to GIF (Like AI SOC Demo)

### Option A: Using agg (Recommended - High Quality)

```bash
# Create demo/recordings directory if it doesn't exist
mkdir -p demo/recordings

# Convert .cast to .gif with optimal settings
agg \
  --cols 120 \
  --rows 40 \
  --speed 1.5 \
  --font-size 14 \
  --theme monokai \
  demo/recordings/threat_hunting_demo.cast \
  demo/recordings/threat_hunting_demo.gif

# For faster playback (2x speed):
agg \
  --cols 120 \
  --rows 40 \
  --speed 2.0 \
  --font-size 14 \
  --theme monokai \
  demo/recordings/threat_hunting_demo.cast \
  demo/recordings/threat_hunting_demo_fast.gif
```

**agg Options Explained:**
- `--cols 120` - Terminal width (120 columns for widescreen)
- `--rows 40` - Terminal height (40 rows, good for demos)
- `--speed 1.5` - Playback speed (1.5x faster than recorded)
- `--font-size 14` - Font size (14 is readable)
- `--theme monokai` - Color theme (monokai, dracula, solarized-dark, etc.)

### Option B: Using asciicast2gif (Alternative)

```bash
# Install asciicast2gif
npm install -g asciicast2gif

# Convert
asciicast2gif \
  -s 1.5 \
  -t monokai \
  demo/recordings/threat_hunting_demo.cast \
  demo/recordings/threat_hunting_demo.gif
```

---

## 🎯 Recording Best Practices

### 1. Terminal Setup

```bash
# Set a clean prompt (optional)
export PS1="\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "

# Or even simpler:
export PS1="$ "

# Use a readable terminal theme
# Recommended: Solarized Dark, Monokai, Dracula
```

### 2. Window Size

**For Widescreen (Recommended):**
```bash
# 120 columns × 40 rows
printf '\e[8;40;120t'
```

**For Standard:**
```bash
# 100 columns × 30 rows
printf '\e[8;30;100t'
```

### 3. Font Size

- Use at least 14pt font
- Monospace font (Monaco, Menlo, Fira Code)
- Ensure text is readable when viewing GIF

### 4. Recording Tips

**Before Recording:**
- [ ] Close unnecessary applications
- [ ] Disable notifications
- [ ] Clear terminal history: `clear`
- [ ] Test run simulation once: `python3 demo_simulation.py`
- [ ] Know when to press ENTER (7 times total)

**During Recording:**
- [ ] Speak/narrate if making video (optional)
- [ ] Don't rush - let output display
- [ ] Press ENTER smoothly at prompts
- [ ] If you make a mistake, stop and re-record

**After Recording:**
- [ ] Review the .cast file: `asciinema play demo/recordings/threat_hunting_demo.cast`
- [ ] If good, convert to GIF
- [ ] If not, delete and re-record

---

## 📐 Optimal Settings for Different Uses

### For GitHub README (Small, Fast)

```bash
agg \
  --cols 100 \
  --rows 30 \
  --speed 2.0 \
  --font-size 12 \
  --theme monokai \
  demo/recordings/threat_hunting_demo.cast \
  demo/recordings/threat_hunting_readme.gif

# Optional: Compress further
# Install gifsicle: brew install gifsicle
gifsicle --optimize=3 --colors 256 \
  demo/recordings/threat_hunting_readme.gif \
  -o demo/recordings/threat_hunting_readme_optimized.gif
```

### For Presentation Slides (Large, Clear)

```bash
agg \
  --cols 120 \
  --rows 40 \
  --speed 1.5 \
  --font-size 16 \
  --theme dracula \
  demo/recordings/threat_hunting_demo.cast \
  demo/recordings/threat_hunting_presentation.gif
```

### For Documentation (Balanced)

```bash
agg \
  --cols 110 \
  --rows 35 \
  --speed 1.75 \
  --font-size 14 \
  --theme monokai \
  demo/recordings/threat_hunting_demo.cast \
  demo/recordings/threat_hunting_docs.gif
```

---

## 🎭 Recording Multiple Versions

### Version 1: Full Demo (All Phases)

```bash
# Record complete simulation
asciinema rec demo/recordings/full_demo.cast

python3 demo_simulation.py
# Press ENTER at all 7 prompts

# Ctrl+D to stop

# Convert
agg --speed 1.5 --theme monokai \
  demo/recordings/full_demo.cast \
  demo/recordings/full_demo.gif
```

### Version 2: Quick Highlights (Auto Mode)

```bash
# Record with auto-progression
asciinema rec demo/recordings/quick_demo.cast

# Auto-advance with yes command
yes "" | python3 demo_simulation.py

# Ctrl+D after it completes

# Convert with faster speed
agg --speed 2.5 --theme monokai \
  demo/recordings/quick_demo.cast \
  demo/recordings/quick_demo.gif
```

### Version 3: Individual Phases

Record each phase separately for focused demonstrations:

```bash
# Record just VALHALLA phase
asciinema rec demo/recordings/phase1_valhalla.cast
# Run simulation, press ENTER after Phase 1, then Ctrl+C
# Ctrl+D to stop recording

# Record just THOR scanning
# Edit demo_simulation.py to skip to Phase 3
# Then record
```

---

## 📊 File Size Management

### Check File Sizes

```bash
# Check .cast file size
ls -lh demo/recordings/*.cast

# Check .gif file size
ls -lh demo/recordings/*.gif
```

### Reduce GIF Size

**Method 1: Lower FPS**
```bash
agg \
  --fps-cap 10 \
  --speed 2.0 \
  demo/recordings/threat_hunting_demo.cast \
  demo/recordings/threat_hunting_small.gif
```

**Method 2: Reduce Colors**
```bash
# Using gifsicle
gifsicle --optimize=3 --colors 128 \
  demo/recordings/threat_hunting_demo.gif \
  -o demo/recordings/threat_hunting_compressed.gif
```

**Method 3: Reduce Dimensions**
```bash
agg \
  --cols 90 \
  --rows 25 \
  --font-size 12 \
  demo/recordings/threat_hunting_demo.cast \
  demo/recordings/threat_hunting_compact.gif
```

**Target Sizes:**
- GitHub README: < 10 MB (smaller is better)
- Documentation: < 20 MB
- Presentation: < 50 MB

---

## 🎨 Theme Options

### Available Themes for agg

```bash
# Light themes
--theme github-light
--theme solarized-light

# Dark themes (Recommended)
--theme monokai          # Purple/pink highlights
--theme dracula          # Purple/pink/cyan
--theme solarized-dark   # Blue/green
--theme nord             # Blue/teal
--theme gruvbox-dark     # Orange/yellow

# Try different themes:
for theme in monokai dracula solarized-dark nord; do
  agg --theme $theme \
    demo/recordings/threat_hunting_demo.cast \
    demo/recordings/demo_${theme}.gif
done
```

---

## 📹 Complete Recording Script

Here's a ready-to-use script for recording:

```bash
#!/bin/bash
# record_demo.sh - Automated recording script

set -e

RECORDINGS_DIR="demo/recordings"
DEMO_NAME="threat_hunting_demo"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create directories
mkdir -p "$RECORDINGS_DIR"

echo "🎬 Starting recording..."
echo "   Recording will be saved to: ${RECORDINGS_DIR}/${DEMO_NAME}_${TIMESTAMP}.cast"
echo ""
echo "📋 Instructions:"
echo "   1. Press ENTER at each phase (7 times total)"
echo "   2. Press Ctrl+D when complete to stop recording"
echo ""
read -p "Press ENTER to start recording..."

# Start recording
asciinema rec "${RECORDINGS_DIR}/${DEMO_NAME}_${TIMESTAMP}.cast"

echo ""
echo "✅ Recording complete!"
echo ""
echo "🎨 Converting to GIF..."

# Convert to GIF with optimal settings
agg \
  --cols 120 \
  --rows 40 \
  --speed 1.5 \
  --font-size 14 \
  --theme monokai \
  "${RECORDINGS_DIR}/${DEMO_NAME}_${TIMESTAMP}.cast" \
  "${RECORDINGS_DIR}/${DEMO_NAME}_${TIMESTAMP}.gif"

echo "✅ GIF created: ${RECORDINGS_DIR}/${DEMO_NAME}_${TIMESTAMP}.gif"

# Get file size
SIZE=$(du -h "${RECORDINGS_DIR}/${DEMO_NAME}_${TIMESTAMP}.gif" | cut -f1)
echo "   File size: $SIZE"

# Play the recording
echo ""
read -p "Press ENTER to preview the recording..."
asciinema play "${RECORDINGS_DIR}/${DEMO_NAME}_${TIMESTAMP}.cast"

echo ""
echo "🎉 All done!"
echo ""
echo "📁 Files created:"
echo "   .cast: ${RECORDINGS_DIR}/${DEMO_NAME}_${TIMESTAMP}.cast"
echo "   .gif:  ${RECORDINGS_DIR}/${DEMO_NAME}_${TIMESTAMP}.gif"
```

**Usage:**
```bash
chmod +x record_demo.sh
./record_demo.sh
```

---

## 🔄 Re-recording (If Needed)

If you make a mistake during recording:

```bash
# Stop current recording
# Press Ctrl+D

# The .cast file is saved, but you can delete it
rm demo/recordings/threat_hunting_demo.cast

# Start fresh
asciinema rec demo/recordings/threat_hunting_demo.cast
```

---

## 📤 Uploading and Sharing

### Option 1: Upload to asciinema.org

```bash
# Upload and get shareable link
asciinema upload demo/recordings/threat_hunting_demo.cast

# You'll get a URL like:
# https://asciinema.org/a/abc123

# Embed in docs:
# [![asciicast](https://asciinema.org/a/abc123.svg)](https://asciinema.org/a/abc123)
```

### Option 2: Host GIF on GitHub

```bash
# Add to git
git add demo/recordings/threat_hunting_demo.gif

# Commit
git commit -m "Add threat hunting simulation demo"

# Push
git push

# Reference in README.md:
# ![Threat Hunting Demo](demo/recordings/threat_hunting_demo.gif)
```

### Option 3: Use in Presentations

- Add .gif to PowerPoint/Keynote slides
- GIFs auto-play in most presentation software
- Ensure file size < 50 MB for smooth playback

---

## 🎯 Quick Reference Commands

```bash
# Record
asciinema rec demo/recordings/demo.cast

# Play back
asciinema play demo/recordings/demo.cast

# Convert to GIF (standard)
agg --speed 1.5 --theme monokai demo/recordings/demo.cast demo/recordings/demo.gif

# Convert to GIF (fast, small)
agg --speed 2.0 --cols 100 --rows 30 --font-size 12 demo/recordings/demo.cast demo/recordings/demo_small.gif

# Compress GIF
gifsicle --optimize=3 --colors 256 demo/recordings/demo.gif -o demo/recordings/demo_optimized.gif

# Upload to asciinema.org
asciinema upload demo/recordings/demo.cast
```

---

## ✅ Recording Checklist

Before recording:
- [ ] Install asciinema and agg
- [ ] Test simulation once
- [ ] Set terminal size (120×40)
- [ ] Choose theme/colors
- [ ] Clear terminal
- [ ] Disable notifications

During recording:
- [ ] Run `asciinema rec demo/recordings/threat_hunting_demo.cast`
- [ ] Run `python3 demo_simulation.py`
- [ ] Press ENTER at 7 prompts
- [ ] Press Ctrl+D when done

After recording:
- [ ] Review: `asciinema play demo/recordings/threat_hunting_demo.cast`
- [ ] Convert to GIF with agg
- [ ] Check file size
- [ ] Optimize if needed
- [ ] Upload/share

---

## 🎬 Example: Full Recording Session

```bash
# Complete example from start to finish

# 1. Navigate to directory
cd /Users/raditio.ghifiardigmail.com/Downloads/ai-driven-soc.backup/

# 2. Create recordings directory
mkdir -p demo/recordings

# 3. Clear terminal
clear

# 4. Start recording
asciinema rec demo/recordings/threat_hunting_demo.cast

# 5. Run simulation (inside the recording)
python3 demo_simulation.py
# [Press ENTER 7 times as prompted]

# 6. Stop recording
# Press Ctrl+D

# 7. Review
asciinema play demo/recordings/threat_hunting_demo.cast

# 8. Convert to GIF
agg \
  --cols 120 \
  --rows 40 \
  --speed 1.5 \
  --font-size 14 \
  --theme monokai \
  demo/recordings/threat_hunting_demo.cast \
  demo/recordings/threat_hunting_demo.gif

# 9. Check size
ls -lh demo/recordings/threat_hunting_demo.gif

# 10. If too large, compress
gifsicle --optimize=3 --colors 256 \
  demo/recordings/threat_hunting_demo.gif \
  -o demo/recordings/threat_hunting_demo_optimized.gif

# 11. Done! Use in README or presentations
```

---

## 🎓 Pro Tips

1. **Practice First**: Run the simulation 2-3 times before recording
2. **Smooth Pacing**: Don't rush through prompts, let output display
3. **Theme Consistency**: Use the same theme as your other demos
4. **File Size**: Aim for < 10 MB for GIFs used in README
5. **Speed**: 1.5x speed is good for demos, 2.0x for quick overviews
6. **Test Playback**: Always review before sharing

---

**Ready to record? Let's create a professional demo!** 🎥

```bash
# Start recording now:
asciinema rec demo/recordings/threat_hunting_demo.cast
python3 demo_simulation.py
```
