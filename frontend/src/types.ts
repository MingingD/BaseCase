export interface LegalCase {
  case_name: string
  category: string
  similarity: number
  snippet: string
  url: string
  /** True when snippet is a TF-IDF-aligned excerpt from the opinion, not a prefix */
  snippet_is_excerpt?: boolean
  /** SVD latent dimensions activated for this query (explainability) */
  why?: string[]
}

/** Classifier + routing; `needs_user_category` prompts pills for low_confidence and no_match */
export interface ClassificationInfo {
  status:
    | 'ok'
    | 'ambiguous'
    | 'low_confidence'
    | 'no_match'
    | 'browse'
    | 'user_selected'
  needs_user_category: boolean
  reason: string | null
  candidates: Array<{ key: string; label: string; score: number }>
}

export interface SearchResponse {
  results: LegalCase[]
  detected_category: string | null
  confidence: number | null
  /** Top latent semantic dimensions for the query (SVD explainability) */
  activated_dimensions?: string[]
  classification?: ClassificationInfo
}
