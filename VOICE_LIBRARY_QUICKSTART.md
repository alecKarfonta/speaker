# Voice Library - Quick Start Guide

## Access the Voice Library

**URL**: http://localhost:3012/voices

The services are running on:
- Frontend: http://localhost:3012
- Backend API: http://localhost:8012

## Quick Start

### 1. Upload Your First Voice

1. Click the **"Add Voice"** button (top right)
2. Enter a voice name (e.g., `my_voice`)
3. Drag and drop one or more `.wav` or `.mp3` files
4. Click **"Upload"**
5. Wait for the upload to complete

### 2. View Voice Details

1. Click on any voice card to expand it
2. You'll see all audio files associated with that voice
3. Each file shows:
   - File name
   - File size
   - Duration
   - Play, Download, and Delete buttons

### 3. Test a Voice

1. Click the **"Test"** button on any voice card
2. A test panel appears in the bottom right
3. Edit the test text if desired
4. Click **"Generate Speech"**
5. The generated audio will play automatically

### 4. Play Voice Samples

1. Expand a voice card
2. Click the **play button** (▶) on any audio file
3. The file will play in the global audio player at the bottom
4. Use the player controls to pause/resume

### 5. Download Voice Files

1. Expand a voice card
2. Click the **download icon** on any file
3. The file will download to your browser's download folder

### 6. Combine Multiple Voices

1. Click **"Combine Voices"** button (top right)
2. Select 2 or more voices by clicking on them (checkboxes appear)
3. Click **"Combine (N)"** where N is the number of selected voices
4. Enter a name for the new combined voice
5. Click OK
6. A new voice is created with all files from the selected voices

### 7. Delete Voices or Files

**Delete entire voice:**
1. Click the menu icon (⋮) on a voice card
2. Click **"Delete Voice"**
3. Confirm the deletion

**Delete individual file:**
1. Expand a voice card
2. Click the **trash icon** on any file
3. Confirm the deletion
4. If this was the last file, the entire voice is removed

### 8. Search Voices

1. Type in the search bar at the top
2. Voices are filtered in real-time by name
3. Clear the search to see all voices again

## Current Voices Available

You have **7 voices** loaded:
- trump_cp
- major
- biden
- dsp
- trump
- loli
- batman

## Tips & Tricks

- **Upload multiple files at once**: Select or drag multiple files when adding a voice
- **Add files to existing voice**: Upload with the same voice name to add more samples
- **Test before committing**: Use the test feature to verify voice quality before using in production
- **Organize by naming**: Use descriptive names with underscores (e.g., `narrator_male_deep`)
- **Back up your voices**: Download important voice files to keep backups

## Keyboard Shortcuts

Currently none implemented, but could add:
- `Ctrl/Cmd + U` - Upload voice
- `Space` - Play/Pause audio
- `Esc` - Close modals
- `Delete` - Delete selected voice

## Troubleshooting

### Upload fails
- Check that files are .wav or .mp3 format
- Try smaller files if upload times out
- Check backend logs: `docker compose logs tts-api`

### Audio won't play
- Check browser permissions for audio
- Try a different browser
- Check file is valid audio format

### Voice doesn't appear
- Click the Refresh button
- Check backend logs for errors
- Verify files exist in `data/voices/` directory

### Backend not responding
```bash
# Check services status
docker compose ps

# Restart backend if needed
docker compose restart tts-api

# View logs
docker compose logs tts-api --tail=50
```

### Frontend not loading
```bash
# Restart frontend
docker compose restart frontend

# View logs
docker compose logs frontend --tail=50
```

## Advanced Usage

### API Direct Access

You can also use the API directly:

```bash
# List voices
curl http://localhost:8012/voices

# Get voice details
curl http://localhost:8012/voices/trump/details

# Upload voice file
curl -X POST "http://localhost:8012/voices?voice_name=my_voice" \
  -F "file=@/path/to/audio.wav"

# Delete voice
curl -X DELETE http://localhost:8012/voices/my_voice

# Combine voices
curl -X POST http://localhost:8012/voices/combine \
  -H "Content-Type: application/json" \
  -d '{"voice_name": "combined", "source_voices": ["voice1", "voice2"]}'

# Generate TTS
curl -X POST http://localhost:8012/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice_name": "trump", "language": "en", "output_format": "wav"}' \
  --output speech.wav
```

## Features Showcase

### What Makes This Voice Library World-Class?

1. **Intuitive UI**: Clean, modern interface that's easy to navigate
2. **Drag & Drop**: Upload files with simple drag and drop
3. **Real-time Feedback**: Instant visual feedback for all operations
4. **Audio Playback**: Built-in player with progress bar
5. **File Management**: Complete control over individual files
6. **Voice Testing**: Test synthesis without leaving the page
7. **Voice Combining**: Merge multiple voices effortlessly
8. **Search & Filter**: Find voices quickly
9. **Responsive**: Works on desktop, tablet, and mobile
10. **Professional Design**: Glassmorphism, gradients, animations

### UI Highlights

- **Animated backgrounds**: Floating gradient orbs
- **Smooth transitions**: Framer Motion animations
- **Glass cards**: Modern glassmorphic design
- **Hover effects**: Interactive states throughout
- **Toast notifications**: Clear success/error messages
- **Loading states**: Visual feedback during operations
- **Expandable cards**: Click to see details
- **Badge indicators**: File count, size, etc.

## What's Next?

Potential enhancements:
- Voice categories and tags
- Usage statistics
- Audio waveform visualization
- Voice quality analysis
- Batch operations
- Export/import voice packs
- Cloud backup
- Voice cloning wizard
- Audio editing tools

## Need Help?

- Check `VOICE_LIBRARY_IMPLEMENTATION.md` for technical details
- View backend logs: `docker compose logs tts-api`
- View frontend logs: `docker compose logs frontend`
- Check browser console for frontend errors

## Performance Notes

- Loading 7 voices: ~1.8GB GPU memory (XTTS backend)
- TTS generation: ~2-5 seconds for short text
- Model loading: ~30 seconds on first start
- Voice upload: Depends on file size
- Current limit: 100 requests/minute per IP

---

**Enjoy your world-class voice library!** 🎙️✨

