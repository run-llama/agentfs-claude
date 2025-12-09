import { LlamaParseReader } from "llama-cloud-services";

const apiKey = process.env.LLAMA_CLOUD_API_KEY;
const reader = new LlamaParseReader({
  resultType: "text",
  apiKey: apiKey,
  fastMode: true,
  checkInterval: 4,
});

export async function parseFile(filePath: string): Promise<string> {
  let text = "";
  try {
    const documents = await reader.loadData(filePath);
    for (const document of documents) {
      text += document.text;
    }
    return text;
  } catch (error) {
    console.log(error);
    return text;
  }
}
