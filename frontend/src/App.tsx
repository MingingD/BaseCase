import { useState } from 'react'
import './App.css'
import SearchIcon from './assets/mag.png'
import { LegalCase, SearchResponse } from './types'

const CATEGORY_COLORS: Record<string, string> = {
  'personal injury': '#e74c3c',
  'employment': '#3498db',
  'copyright': '#2ecc71',
}

function categoryColor(cat: string): string {
  for (const [key, color] of Object.entries(CATEGORY_COLORS)) {
    if (cat.toLowerCase().includes(key)) return color
  }
  return '#95a5a6'
}

function App(): JSX.Element {
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [results, setResults] = useState<LegalCase[]>([])
  const [detectedCategory, setDetectedCategory] = useState<string | null>(null)
  const [confidence, setConfidence] = useState<number | null>(null)

  const handleSearch = async (value: string): Promise<void> => {
    setSearchTerm(value)
    if (value.trim() === '') {
      setResults([])
      setDetectedCategory(null)
      setConfidence(null)
      return
    }
    const response = await fetch(`/api/search?q=${encodeURIComponent(value)}`)
    const data: SearchResponse = await response.json()
    setResults(data.results)
    setDetectedCategory(data.detected_category)
    setConfidence(data.confidence)
  }

  return (
    <div className="full-body-container">
      <div className="top-text">
        {/* Title */}
        <div className="google-colors">
          <h1 id="google-4">B</h1>
          <h1 id="google-3">a</h1>
          <h1 id="google-0-1">s</h1>
          <h1 id="google-0-2">e</h1>
          <h1 id="google-4">C</h1>
          <h1 id="google-3">a</h1>
          <h1 id="google-0-1">s</h1>
          <h1 id="google-0-2">e</h1>
        </div>

        {/* Disclaimer */}
        <div className="disclaimer-banner">
          ⚠️ This tool provides legal information only — not formal legal advice.
        </div>

        {/* Category pills */}
        <div className="category-pills">
          <span className="pill pill-injury">Personal Injury</span>
          <span className="pill pill-employment">Employment Law</span>
          <span className="pill pill-copyright">Copyright</span>
        </div>

        {/* Search box */}
        <div className="input-box" onClick={() => document.getElementById('search-input')?.focus()}>
          <img src={SearchIcon} alt="search" />
          <input
            id="search-input"
            placeholder="Describe your legal situation…"
            value={searchTerm}
            onChange={(e) => handleSearch(e.target.value)}
          />
        </div>
      </div>

      {/* Detected category */}
      {detectedCategory && (
        <div className="detected-category">
          Detected area: <strong>{detectedCategory}</strong>
          {confidence !== null && (
            <span className="confidence"> (confidence: {(confidence * 100).toFixed(0)}%)</span>
          )}
        </div>
      )}

      {/* Results */}
      <div id="answer-box">
        {results.map((c, i) => (
          <div key={i} className="episode-item">
            <h3 className="episode-title">{c.case_name}</h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
              <span
                className="category-badge"
                style={{ backgroundColor: categoryColor(c.category) }}
              >
                {c.category}
              </span>
              <span className="similarity-score">Match: {(c.similarity * 100).toFixed(0)}%</span>
            </div>
            <p className="episode-desc">{c.snippet}</p>
            {c.url && (
              <a href={c.url} target="_blank" rel="noopener noreferrer" className="courtlistener-link">
                View on CourtListener →
              </a>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

export default App
