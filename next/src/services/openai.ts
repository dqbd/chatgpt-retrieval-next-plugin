import { ChatCompletionRequestMessage, type OpenAIApi } from "openai"

// TODO: Implement exponential backoff

async function getEmbeddings(
  openai: OpenAIApi,
  input: string[]
): Promise<number[][]> {
  const response = await openai.createEmbedding({
    model: "text-embedding-ada-002",
    input,
  })

  return response.data.data.map((i) => i.embedding)
}
export async function getChatCompletion(
  openai: OpenAIApi,
  messages: ChatCompletionRequestMessage[],
  model: string = "gpt-3.5-turbo"
) {
  const response = await openai.createChatCompletion({
    model,
    messages,
  })

  const completion = response.data.choices[0].message?.content.trim()
  console.log("Completion:", completion)
  return completion
}
