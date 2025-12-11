#!/bin/bash
# Full GLM-TTS Parameter Sweep

TTS_API="http://localhost:8016"
STT_API="http://localhost:8603/v1/audio/transcriptions"

# Test sentences
SENTENCES=(
    "Hello world."
    "The quick brown fox jumps over the lazy dog."
    "Welcome to the demonstration of text to speech."
)

# Function to test current config
test_config() {
    local config_name="$1"
    local total_correct=0
    local total_tests=0
    
    echo "Testing: $config_name"
    
    for sentence in "${SENTENCES[@]}"; do
        # Generate TTS
        curl -s -X POST "$TTS_API/tts" -H "Content-Type: application/json" \
            -d "{\"text\": \"$sentence\", \"voice_name\": \"trump\", \"language\": \"en\", \"output_format\": \"wav\"}" \
            -o /tmp/sweep_test.wav
        
        # Transcribe
        result=$(curl -s "$STT_API" \
            -H "Authorization: Bearer stt-api-key" \
            -F file="@/tmp/sweep_test.wav" \
            -F model="base" \
            -F language="en" | jq -r '.text // "ERROR"')
        
        # Simple word match count
        orig_words=$(echo "$sentence" | tr '[:upper:]' '[:lower:]' | tr -d '.,!?' | wc -w)
        match_count=0
        for word in $(echo "$sentence" | tr '[:upper:]' '[:lower:]' | tr -d '.,!?'); do
            if echo "$result" | tr '[:upper:]' '[:lower:]' | grep -qw "$word"; then
                ((match_count++))
            fi
        done
        
        accuracy=$((match_count * 100 / orig_words))
        ((total_tests++))
        if [ $accuracy -ge 80 ]; then
            ((total_correct++))
            echo "  ✅ $accuracy% | $sentence"
        else
            echo "  ❌ $accuracy% | $sentence → $result"
        fi
    done
    
    avg=$((total_correct * 100 / total_tests))
    echo "  Overall: $total_correct/$total_tests passed ($avg%)"
    echo ""
}

echo "========================================"
echo "GLM-TTS Parameter Sweep Results"
echo "========================================"
echo ""

# Test current config
test_config "CURRENT (sampling=25, min_ratio=8, max_ratio=30)"

echo "To test other configs, we need to restart the container with different env vars."
echo "Example configs to test:"
echo ""
echo "| Config | sampling | min_ratio | max_ratio | Expected |"
echo "|--------|----------|-----------|-----------|----------|"
echo "| Current | 25 | 8 | 30 | Baseline |"
echo "| Low sampling | 10 | 8 | 30 | More deterministic |"
echo "| High sampling | 50 | 8 | 30 | More variation |"
echo "| Higher min_ratio | 25 | 12 | 30 | Longer output |"
echo "| Higher max_ratio | 25 | 8 | 50 | Allow longer |"
