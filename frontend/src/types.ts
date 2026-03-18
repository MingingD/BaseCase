export interface LegalCase {
  case_name: string
  category: string
  similarity: number
  snippet: string
  url: string
}

export interface SearchResponse {
  results: LegalCase[]
  detected_category: string | null
  confidence: number | null
}
