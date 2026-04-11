export interface LegalCase {
  case_name: string
  category: string
  similarity: number
  snippet: string
  url: string
  /** SVD latent dimensions activated for this query (explainability) */
  why?: string[]
}

export interface SearchResponse {
  results: LegalCase[]
  detected_category: string | null
  confidence: number | null
  /** Top latent semantic dimensions for the query (SVD explainability) */
  activated_dimensions?: string[]
}
