import { useState, useEffect } from 'react'
import { Play } from 'lucide-react'
import { toast, Toaster } from 'react-hot-toast'

interface Voice {
  id: string;
  name: string;
}

interface TTSResponse {
  id: number;
  text: string;
  voice: string;
  audioUrl: string;
  timestamp: string;
}

function App() {
  const [text, setText] = useState('')
  const [voiceList, setVoiceList] = useState<Voice[]>([])
  const [selectedVoice, setSelectedVoice] = useState('')
  const [responses, setResponses] = useState<TTSResponse[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetchVoices()
  }, [])

  const fetchVoices = async () => {
    try {
      const response = await fetch('/api/voices')
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const responseData = await response.json()
      const voiceNames = responseData.voices || responseData
      
      const voices: Voice[] = voiceNames.map((name: string) => ({
        id: name,
        name: name
      }))
      
      setVoiceList(voices)
      if (voices.length > 0 && !selectedVoice) {
        setSelectedVoice(voices[0].name)
      }
    } catch (error) {
      console.error('Error fetching voices:', error)
      toast.error(`Failed to fetch voices: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  const handleSubmit = async () => {
    if (!text.trim() || !selectedVoice) {
      toast.error('Please enter text and select a voice')
      return
    }

    setLoading(true)

    try {
      const response = await fetch('/api/tts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: text,
          voice_name: selectedVoice,
          language: 'en'
        })
      })

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}: ${await response.text()}`)
      }

      const buffer = await response.arrayBuffer()
      const audioUrl = URL.createObjectURL(new Blob([buffer], { type: 'audio/wav' }))

      const newResponse: TTSResponse = {
        id: Date.now(),
        text,
        voice: selectedVoice,
        audioUrl,
        timestamp: new Date().toLocaleTimeString()
      }

      setResponses(prev => [newResponse, ...prev])
      toast.success('Audio generated successfully!')

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred'
      toast.error(`Failed to generate speech: ${errorMessage}`)
      console.error('Error generating speech:', error)
    } finally {
      setLoading(false)
    }
  }

  const playAudio = (audioUrl: string) => {
    const audio = new Audio(audioUrl)
    audio.play().catch(err => {
      console.error('Error playing audio:', err)
      toast.error('Failed to play audio')
    })
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <Toaster position="top-right" />
      
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-center mb-8">Speaker TTS API</h1>
        
        <div className="bg-gray-800 rounded-lg p-6 mb-6">
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Voice</label>
            <select
              value={selectedVoice}
              onChange={(e) => setSelectedVoice(e.target.value)}
              className="w-full p-3 bg-gray-700 border border-gray-600 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {voiceList.map((voice) => (
                <option key={voice.id} value={voice.name}>
                  {voice.name}
                </option>
              ))}
            </select>
          </div>

          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Text to Speech</label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter text to convert to speech..."
              className="w-full p-3 bg-gray-700 border border-gray-600 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              rows={4}
            />
          </div>

          <button
            onClick={handleSubmit}
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white font-medium py-3 px-4 rounded-md transition-colors flex items-center justify-center"
          >
            {loading ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                Generating...
              </>
            ) : (
              <>
                <Play className="w-5 h-5 mr-2" />
                Generate Speech
              </>
            )}
          </button>
        </div>

        {responses.length > 0 && (
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-2xl font-bold mb-4">Generated Audio</h2>
            <div className="space-y-4">
              {responses.map((response) => (
                <div key={response.id} className="bg-gray-700 rounded-lg p-4">
                  <div className="flex justify-between items-start mb-2">
                    <div>
                      <p className="font-medium">{response.voice}</p>
                      <p className="text-gray-300 text-sm">{response.timestamp}</p>
                    </div>
                    <button
                      onClick={() => playAudio(response.audioUrl)}
                      className="bg-green-600 hover:bg-green-700 text-white p-2 rounded-md transition-colors"
                    >
                      <Play className="w-4 h-4" />
                    </button>
                  </div>
                  <p className="text-gray-300">{response.text}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App 