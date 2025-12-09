import * as readline from "readline/promises";

export async function consoleInput(question: string): Promise<string> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const answer = await rl.question(question);
  rl.close();
  return answer;
}
