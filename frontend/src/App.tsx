import { useState, useEffect } from 'react'
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

const PILL_CATEGORIES = [
  { label: 'Personal Injury', key: 'personal_injury',  cls: 'injury' },
  { label: 'Employment Law',  key: 'employment_labor', cls: 'employment' },
  { label: 'Copyright',       key: 'copyright',        cls: 'copyright' },
]

function App(): JSX.Element {
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [results, setResults] = useState<LegalCase[]>([])
  const [detectedCategory, setDetectedCategory] = useState<string | null>(null)
  const [confidence, setConfidence] = useState<number | null>(null)
  const [activatedDimensions, setActivatedDimensions] = useState<string[]>([])
  const [activeCategory, setActiveCategory] = useState<string | null>(null)

  const fetchResults = async (q: string, category: string | null): Promise<void> => {
    const params = new URLSearchParams()
    if (q.trim()) params.set('q', q)
    if (category) params.set('category', category)
    const response = await fetch(`/api/search?${params.toString()}`)
    const data: SearchResponse = await response.json()
    setResults(data.results)
    setDetectedCategory(data.detected_category)
    setConfidence(data.confidence)
    setActivatedDimensions(data.activated_dimensions ?? [])
  }

  useEffect(() => { fetchResults('', null) }, [])

  const handleSearch = (value: string): void => {
    setSearchTerm(value)
    fetchResults(value, activeCategory)
    if (!value.trim()) {
      setDetectedCategory(null)
      setActivatedDimensions([])
    }
  }

  const handlePillClick = (key: string): void => {
    const next = activeCategory === key ? null : key
    setActiveCategory(next)
    fetchResults(searchTerm, next)
  }

  return (
    <>
      <header className="site-header">
        <span className="site-title">BaseCase</span>
      </header>

      <main className="main-content">
        <div className="disclaimer-banner">
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{verticalAlign:'middle', marginRight:'0.4rem', flexShrink:0}}>
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
            <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
          </svg>
          This tool provides legal information only — not formal legal advice.
        </div>

        <div className="category-pills">
          {PILL_CATEGORIES.map(({ label, key, cls }) => (
            <span
              key={key}
              className={`pill pill-${cls}${activeCategory === key ? ' pill-active' : ''}`}
              onClick={() => handlePillClick(key)}
            >
              {label}
            </span>
          ))}
        </div>

        <div className="search-row">
          <img src={SearchIcon} alt="" className="search-icon" />
          <input
            placeholder="Describe your legal situation…"
            value={searchTerm}
            onChange={(e) => handleSearch(e.target.value)}
            onPaste={(e) => {
              setTimeout(() => handleSearch((e.target as HTMLInputElement).value), 0)
            }}
            autoFocus
          />
        </div>

        {detectedCategory && (
          <p className="detected-category">
            detected area: <strong>{detectedCategory}</strong>
            {confidence !== null && (
              <span className="confidence"> — {(confidence * 100).toFixed(0)}% confidence</span>
            )}
          </p>
        )}

        {activatedDimensions.length > 0 && (
          <div className="query-explainability" aria-label="Query latent dimensions">
            <span className="query-explainability-label">Query matches these themes (SVD):</span>
            <ul className="query-explainability-list">
              {activatedDimensions.map((dim, j) => (
                <li key={j}>{dim}</li>
              ))}
            </ul>
          </div>
        )}

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
                {searchTerm.trim() && (
                  <p className="snippet-label">EXCERPT ALIGNED TO YOUR SEARCH</p>
                )}
                <p className="result-snippet">{c.snippet}</p>
                {c.why && c.why.length > 0 && (
                  <div className="why-this-result">
                    <span className="why-label">Why this result?</span>
                    <ul className="why-list">
                      {c.why.map((line, k) => (
                        <li key={k}>{line}</li>
                      ))}
                    </ul>
                  </div>
                )}
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
