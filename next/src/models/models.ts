export const Source = {
  email: "email",
  file: "file",
  chat: "chat",
} as const

export type Source = typeof Source[keyof typeof Source]

export interface DocumentMetadata {
  source?: Source
  sourceId?: string
  url?: string
  createAt?: string
  author?: string
}

export interface DocumentChunkMetadata extends DocumentMetadata {
  documentId?: string
}

export interface DocumentChunk {
  id?: string
  text: string
  metadata: DocumentChunkMetadata
  embedding?: number[]
}

export interface DocumentChunkWithScore extends DocumentChunk {
  score: number
}

export interface Document {
  id?: string
  text: string
  metadata?: DocumentMetadata
}

export interface DocumentWithChunks extends Document {
  chunks: DocumentChunk[]
}

export interface DocumentMetadataFilter {
  documentId?: string
  source?: Source
  sourceId?: string
  author?: string
  startDate?: string
  endDate?: string
}

export interface Query {
  query: string
  filter: DocumentMetadataFilter
  topK?: number // TODO: default to 3
}

export interface QueryWithEmbedding extends Query {
  embedding: number[]
}

export interface QueryResult {
  query: string
  results: DocumentChunkWithScore[]
}
