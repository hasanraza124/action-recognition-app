import { useState } from 'react'
import './App.css'

function App() {
  const [selectedImage, setSelectedImage] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleImageChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setSelectedImage(file)
      setPreviewUrl(URL.createObjectURL(file))
      setResult(null)
    }
  }

  const handleClear = () => {
    setSelectedImage(null)
    setPreviewUrl(null)
    setResult(null)
  }

  const handlePredict = async () => {
    if (!selectedImage) return

    setLoading(true)
    setResult(null)

    const formData = new FormData()
    formData.append('image', selectedImage)

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      // Check if response has the expected format
      if (!data.actions || !data.caption) {
        console.error("Unexpected response format:", data)
        alert("Backend returned an unexpected format. Please make sure you have restarted the backend server to apply the latest changes.")
        return
      }

      setResult(data)
    } catch (error) {
      console.error("Error predicting:", error)
      alert(error.message || "Failed to get prediction. Is the backend running?")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen w-full flex flex-col items-center p-4 md:p-8">
      <div className="hexagon-bg"></div>
      
      {/* Header */}
      <header className="text-center mb-10 w-full animate-in fade-in duration-700">
        <h1 className="text-lg md:text-4xl font-bold tracking-widest text-white uppercase mb-1">
          Action Recognition & Captioning
        </h1>
      </header>

      <main className="grid grid-cols-1 lg:grid-cols-[1.2fr_0.8fr] gap-10 w-full max-w-[1400px] flex-1 items-start">
        {/* Left Column*/}
        <div className="flex flex-col gap-6 h-full">
          <div className="bg-panel neon-border-panel rounded-[2rem] p-8 md:p-10 flex flex-col items-center justify-between min-h-[500px] flex-1">
            <div className="text-center mb-8">
              <h2 className="text-4xl md:text-3xl font-black mb-2 text-white uppercase tracking-wide px-40">
                Upload Image
              </h2>
            </div>

            {/* Upload Area */}
            <div 
              className={`relative w-full aspect-[4/3] rounded-[2rem] dashed-glow flex items-center justify-center overflow-hidden transition-all duration-300 group ${!previewUrl ? 'cursor-pointer hover:bg-white/5' : ''}`}
              onClick={() => !previewUrl && document.getElementById('file-upload').click()}
            >
              {previewUrl ? (
                <div className="relative w-full h-full">
                  <img 
                    src={previewUrl} 
                    alt="Preview" 
                    className="w-full h-full object-cover" 
                  />
                  <button 
                    className="absolute top-4 right-4 bg-black/50 text-white rounded-full p-2 hover:bg-black/80 transition-colors"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleClear();
                    }}
                  >
                    ✕
                  </button>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-4">
                  <div className="w-16 h-16 md:w-20 md:h-20 border-2 border-[#38bdf8] rounded-2xl flex items-center justify-center transition-transform group-hover:scale-110">
                    <svg className="w-8 h-8 md:w-10 md:h-10 text-[#38bdf8]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                    </svg>
                  </div>
                  <div className="text-center">
                    <p className="text-[#38bdf8] font-bold text-lg">Click to Upload Image</p>
                    <p className="text-[#38bdf8]/60 text-sm">or Drag & Drop</p>
                  </div>
                </div>
              )}
              <input
                type="file"
                accept="image/*"
                onChange={handleImageChange}
                id="file-upload"
                className="hidden"
              />
            </div>

            {/* Action Buttons */}
            <div className="grid grid-cols-2 gap-4 w-full mt-10">
              <button
                className="btn-clear py-4 rounded-xl font-bold uppercase tracking-widest text-[#94a3b8] text-xs md:text-sm border border-white/5 disabled:opacity-30"
                onClick={handleClear}
                disabled={!previewUrl || loading}
              >
                Clear
              </button>
              <button
                className="btn-analyze py-4 rounded-xl font-bold uppercase tracking-widest text-white text-xs md:text-sm border border-white/10 disabled:opacity-30"
                onClick={handlePredict}
                disabled={!selectedImage || loading}
              >
                {loading ? 'Analyzing...' : 'Analyze'}
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Results */}
        <div className="flex flex-col gap-6 h-full">
          {/* Predicted Actions */}
          <div className="bg-panel rounded-[1.5rem] border border-white/5 flex flex-col min-h-[300px]">
            <div className="px-6 py-4 border-b border-white/5">
              <h4 className="text-[#888] font-semibold text-[10px] tracking-[0.2em] uppercase">
                Predicted Actions
              </h4>
            </div>
            <div className="p-8 flex flex-col gap-4 flex-1 justify-center">
              {result ? (
                result.actions.slice(0, 3).map((action, index) => (
                  <div key={index} className="flex justify-start animate-in slide-in-from-left duration-500" style={{ animationDelay: `${index * 100}ms` }}>
                    <div className={`progress-container chip-${index}`}>
                      <div 
                        className={`progress-fill fill-${index}`} 
                        style={{ width: `${Math.round((action.confidence || action.score || 0) * 100)}%` }}
                      ></div>
                      <div className="progress-content">
                        <div className="flex items-center gap-2">
                          <span className="capitalize">{action.action || action.label}</span>
                          <span className="opacity-60 text-sm">({Math.round((action.confidence || action.score || 0) * 100)}%)</span>
                        </div>
                        <div className="w-5 h-5 rounded-full border border-current flex items-center justify-center opacity-40">
                          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                          </svg>
                        </div>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="flex flex-col items-center justify-center opacity-20 text-center py-10">
                  <p className="text-xs uppercase tracking-widest">Awaiting Analysis</p>
                </div>
              )}
            </div>
          </div>

          {/* Detected Actions / Caption */}
          <div className="bg-panel rounded-[1.5rem] border border-white/5 flex flex-col min-h-[250px] relative">
            <div className="px-6 py-4 border-b border-white/5">
              <h4 className="text-[#888] font-semibold text-[10px] tracking-[0.2em] uppercase">
                Detected Actions
              </h4>
            </div>
            <div className="p-8 flex flex-col flex-1">
              <h5 className="text-white font-bold text-sm md:text-base mb-2 uppercase italic tracking-tight">
                Caption
              </h5>
              {result ? (
                <p className="text-[#aaa] italic leading-relaxed text-sm md:text-base">
                   {result.caption}
                </p>
              ) : (
                <p className="text-[#555] italic text-sm">
                   Caption will appear here...
                </p>
              )}
                            
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App

