import { type OpenAIApi } from "openai"
import { outdent } from "outdent"
import { getChatCompletion } from "./openai"

export async function screenTextForPii(
  content: string,
  openai: OpenAIApi
): Promise<boolean> {
  const completion = await getChatCompletion(openai, [
    {
      role: "system",
      content: outdent`
        You can only respond with the word "True" or "False", where your answer indicates whether the text in the user's message contains PII.
        Do not explain your answer, and do not use punctuation.
        Your task is to identify whether the text extracted from your company files
        contains sensitive PII information that should not be shared with the broader company. Here are some things to look out for:
        - An email address that identifies a specific person in either the local-part or the domain
        - The postal address of a private residence (must include at least a street name)
        - The postal address of a public place (must include either a street name or business name)
        - Notes about hiring decisions with mentioned names of candidates. The user will send a document for you to analyze.`,
    },
    { role: "user", content },
  ])

  return !!completion?.includes("True")
}
