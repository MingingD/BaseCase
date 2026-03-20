import { useState } from 'react'
import './App.css'
import SearchIcon from './assets/mag.png'
import { LegalCase, SearchResponse } from './types'

function categoryClass(cat: string): string {
  const c = cat.toLowerCase()
  if (c.includes('personal injury') || c.includes('personal_injury')) return 'injury'
  if (c.includes('employment') || c.includes('employment_labor')) return 'employment'
  if (c.includes('copyright')) return 'copyright'
  return 'default'
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
    <>
      {/* Header */}
      <header className="site-header">
        <span className="site-title">BaseCase</span>
      </header>

      <main className="main-content">
        {/* Disclaimer */}
        <div className="disclaimer-banner">
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{verticalAlign:'middle', marginRight:'0.4rem', flexShrink:0}}>
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
            <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
          This tool provides legal information only — not formal legal advice.
        </div>

        {/* Category pills */}
        <div className="category-pills">
          <span className="pill pill-injury">Personal Injury</span>
          <span className="pill pill-employment">Employment Law</span>
          <span className="pill pill-copyright">Copyright</span>
        </div>

        {/* Search */}
        <div className="search-row">
          <img src={SearchIcon} alt="" className="search-icon" />
          <input
            placeholder="Describe your legal situation…"
            value={searchTerm}
            onChange={(e) => handleSearch(e.target.value)}
            autoFocus
          />
        </div>

        {/* Detected category */}
        {detectedCategory && (
          <p className="detected-category">
            detected area: <strong>{detectedCategory}</strong>
            {confidence !== null && (
              <span className="confidence"> — {(confidence * 100).toFixed(0)}% confidence</span>
            )}
          </p>
        )}

        {/* Results */}
        <div className="results-list">
          {results.map((c, i) => {
            const cat = categoryClass(c.category)
            return (
              <div key={i} className={`result-card cat-${cat}`}>
                <div className="result-meta">
                  <span className={`category-badge badge-${cat}`}>{c.category.replace('_', ' ')}</span>
                  <span className="similarity-score">match: {(c.similarity * 100).toFixed(0)}%</span>
                </div>
                <h3 className="result-title">{c.case_name}</h3>
                <p className="result-snippet">{c.snippet}</p>
                {c.url && (
                  <a href={c.url} target="_blank" rel="noopener noreferrer" className="result-link">
                    view on CourtListener →
                  </a>
                )}
              </div>
            )
          })}
        </div>
      </main>
    </>
  )
}

export default App
