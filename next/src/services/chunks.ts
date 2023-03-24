import { Document, DocumentChunk, DocumentChunkMetadata } from "@/models/models"
import { get_encoding } from "@dqbd/tiktoken"
import { OpenAIApi } from "openai"
import { getEmbeddings } from "@/services/openai"
import { v4 as uuid } from "uuid"

const tokenizer = get_encoding("cl100k_base")
const decoder = new TextDecoder()

// Constants
const CHUNK_SIZE = 200 // The target size of each text chunk in tokens
const MIN_CHUNK_SIZE_CHARS = 350 // The minimum size of each text chunk in characters
const MIN_CHUNK_LENGTH_TO_EMBED = 5 // Discard chunks shorter than this
const EMBEDDINGS_BATCH_SIZE = 128 // The number of embeddings to request at a time
const MAX_NUM_CHUNKS = 10000 // The maximum number of chunks to generate from a text

/**
 * Split a text into chunks of ~CHUNK_SIZE tokens, based on punctuation and newline boundaries.
 * @param text Text to split into chunks
 * @param chunkSize The target size of each chunk in tokens, or None to use the default CHUNK_SIZE.
 * @returns A list of text chunks, each of which is a string of ~CHUNK_SIZE tokens.
 */
export function getTextChunks(
  text: string,
  chunkSize: number = CHUNK_SIZE
): string[] {
  if (!text.trim().length) return []

  // TODO: disallowed_special=()
  let tokens = [...tokenizer.encode(text)]

  const chunks = []
  let numChunks = 0

  while (tokens.length && chunks.length < MAX_NUM_CHUNKS) {
    const chunk = tokens.slice(0, chunkSize)
    let chunkText = decoder.decode(tokenizer.decode(new Uint32Array(chunk)))

    if (!chunkText.trim().length) {
      tokens = tokens.slice(chunk.length)
      continue
    }

    const lastPunctuationIndex = Math.max(
      chunkText.lastIndexOf("."),
      chunkText.lastIndexOf("?"),
      chunkText.lastIndexOf("!"),
      chunkText.lastIndexOf("\n")
    )

    if (
      lastPunctuationIndex > -1 &&
      lastPunctuationIndex > MIN_CHUNK_SIZE_CHARS
    ) {
      chunkText = chunkText.substring(0, lastPunctuationIndex + 1)
    }

    const chunkTextToAppend = chunkText.replaceAll("\n", " ").trim()
    if (chunkTextToAppend.length > MIN_CHUNK_LENGTH_TO_EMBED) {
      chunks.push(chunkTextToAppend)
    }

    // TODO: disallowed_special=()
    tokens = tokens.slice(tokenizer.encode(chunkText).length)
    numChunks += 1
  }

  if (tokens.length > 0) {
    const remainingText = decoder
      .decode(tokenizer.decode(new Uint32Array(tokens)))
      .replaceAll("\n", " ")
      .trim()

    if (remainingText.length > MIN_CHUNK_LENGTH_TO_EMBED) {
      chunks.push(remainingText)
    }
  }

  return chunks
}

/**
 * Create a list of document chunks from a document object and return the document id.
 *
 * @param doc The document object to create chunks from. It should have a text attribute and optionally an id and a metadata attribute.
 * @param chunkSize The target size of each chunk in tokens, or None to use the default CHUNK_SIZE.
 * @returns A tuple of (doc_chunks, doc_id), where doc_chunks is a list of document chunks, each of which is a DocumentChunk object with an id, a document_id, a text, and a metadata attribute, and doc_id is the id of the document object, generated if not provided. The id of each chunk is generated from the document id and a sequential number, and the metadata is copied from the document object.
 */
function createDocumentChunks(
  doc: Document,
  chunkSize?: number
): [DocumentChunk[], string] {
  const docId = doc.id ?? uuid()
  if (!doc.text.trim().length) [[], docId]

  const textChunks = getTextChunks(doc.text, chunkSize)

  const metadata: DocumentChunkMetadata = {
    ...doc.metadata,
    documentId: docId,
  }

  const docChunks: DocumentChunk[] = []

  textChunks.forEach((text, i) => {
    const id = `${docId}_${i}`
    docChunks.push({ id, text, metadata })
  })

  return [docChunks, docId]
}

/**
 * Convert a list of documents into a dictionary from document id to list of document chunks.
 *
 * @param documents The list of documents to convert.
 * @param chunkSize The target size of each chunk in tokens, or None to use the default CHUNK_SIZE.
 * @returns A dictionary mapping each document id to a list of document chunks, each of which is a DocumentChunk object with text, metadata, and embedding attributes.
 */
export async function getDocumentChunks(
  openai: OpenAIApi,
  documents: Document[],
  chunkSize?: number
): Promise<Record<string, DocumentChunk[]>> {
  const chunks: Record<string, DocumentChunk[]> = {}

  const allChunks: DocumentChunk[] = []

  for (const doc of documents) {
    const [docChunks, docId] = await createDocumentChunks(doc, chunkSize)
    allChunks.push(...docChunks)

    chunks[docId] = docChunks
  }

  if (!allChunks.length) return {}

  // Get all the embeddings for the document chunks in batches, using get_embeddings
  const embeddings: number[][] = []

  // Get all the embeddings for the document chunks in batches, using getEmbeddings
  for (let i = 0; i < allChunks.length; i += EMBEDDINGS_BATCH_SIZE) {
    const batchTexts = allChunks.slice(i, i + EMBEDDINGS_BATCH_SIZE)
    const batchEmbeddings = await getEmbeddings(
      openai,
      batchTexts.map((chunk) => chunk.text)
    )
    embeddings.push(...batchEmbeddings)
  }

  // Update the document chunk objects with the embeddings
  allChunks.forEach((chunk, i) => {
    // Assume the object reference to document chunk in chunks and allChunks is the same
    chunk.embedding = embeddings[i]
  })

  return chunks
}
