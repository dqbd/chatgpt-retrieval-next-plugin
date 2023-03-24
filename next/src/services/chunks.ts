import { get_encoding } from "@dqbd/tiktoken"

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
export async function getTextChunks(
  text: string,
  chunkSize: number = CHUNK_SIZE
): Promise<string[]> {
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
