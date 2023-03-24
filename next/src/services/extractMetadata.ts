import { Source } from "@/models/models"
import { OpenAIApi } from "openai"
import { outdent } from "outdent"
import { getChatCompletion } from "@/services/openai"

export async function extractMetadataFromDocument(
  openai: OpenAIApi,
  content: string
): Promise<Record<string, string>> {
  const sources = Object.keys(Source).join(", ")

  const completion = await getChatCompletion(openai, [
    {
      role: "system",
      content: outdent`
        Given a document from a user, try to extract the following metadata:
        - source: string, one of ${sources}
        - url: string or don't specify
        - created_at: string or don't specify
        - author: string or don't specify

        Respond with a JSON containing the extracted metadata in key value pairs. If you don't find a metadata field, don't specify it.`,
    },
    { role: "user", content: content },
  ])

  console.log("Completion: ", completion)

  try {
    return JSON.parse(completion!)
  } catch {
    return {}
  }
}
