import { getDocumentChunks } from "@/services/chunks"
import { getEmbeddings } from "@/services/openai"
import { OpenAIApi } from "openai"
import {
  Document,
  DocumentChunk,
  DocumentMetadataFilter,
  Query,
  QueryResult,
  QueryWithEmbedding,
} from "@/models/models"

export abstract class Datastore {
  constructor(protected openai: OpenAIApi) {}

  /**
   * Takes in a list of documents and inserts them into the database.
   * First deletes all the existing vectors with the document id (if necessary, depends on the vector db), then inserts the new ones.
   * @returns Return a list of document ids.
   */
  async upsert(documents: Document[], chunkSize?: number) {
    // Delete any existing vectors for documents with the input document ids
    await Promise.all(
      documents
        .filter((i) => i.id != null)
        .map((i) =>
          this.delete({ filter: { documentId: i.id }, deleteAll: false })
        )
    )

    const chunks = await getDocumentChunks(this.openai, documents, chunkSize)
    return this._upsert({ chunks })
  }

  /**
   * Takes in a list of list of document chunks and inserts them into the database.
   * Return a list of document ids.
   */
  abstract _upsert(options: {
    chunks: Record<string, DocumentChunk[]>
  }): Promise<string[]>

  /**
   * Takes in a list of queries and filters and returns a list of query results with matching document chunks and scores.
   */
  async query(queries: Query[]): Promise<QueryResult[]> {
    const queryTexts = queries.map((i) => i.query)
    const queryEmbeddings = await getEmbeddings(this.openai, queryTexts)
    const queriesWithEmbeddings = queries.map((query, idx) => ({
      ...query,
      embedding: queryEmbeddings[idx],
    }))

    return await this._query(queriesWithEmbeddings)
  }

  /**
   * Takes in a list of queries with embeddings and filters and returns a list of query results with matching document chunks and scores.
   */
  abstract _query(queries: QueryWithEmbedding[]): Promise<QueryResult[]>

  /**
   * Removes vectors by ids, filter, or everything in the datastore.
   * Multiple parameters can be used at once.
   * Returns whether the operation was successful.
   */
  abstract delete(options: {
    ids?: string[]
    filter?: DocumentMetadataFilter
    deleteAll?: boolean
  }): Promise<void>
}
