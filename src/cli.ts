import * as readline from "readline/promises";
import figlet from "figlet";
import { magentaBright, bold, gray } from "@visulima/colorize";

export async function consoleInput(question: string): Promise<string> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const answer = await rl.question(bold(question));
  rl.close();
  return answer;
}

export async function renderLogo(): Promise<void> {
  const logoText = figlet.textSync("agentfs", {
    font: "ANSI Shadow",
    horizontalLayout: "default",
    verticalLayout: "default",
    width: 100,
    whitespaceBreak: true,
  });

  const logoTextDash = figlet.textSync("-", {
    font: "ANSI Shadow",
    horizontalLayout: "default",
    verticalLayout: "default",
    width: 100,
    whitespaceBreak: true,
  });

  const logoTextClaude = figlet.textSync("claude", {
    font: "ANSI Shadow",
    horizontalLayout: "default",
    verticalLayout: "default",
    width: 100,
    whitespaceBreak: true,
  });

  // Add some styling with picocolors
  const styledLogo = bold(magentaBright(logoText));
  const styledLogoDash = bold(magentaBright(logoTextDash));
  const styledLogoClaude = bold(magentaBright(logoTextClaude));

  // Add some padding/margin
  console.log("\n");
  console.log(styledLogo);
  console.log(styledLogoDash);
  console.log(styledLogoClaude);
  console.log(gray("─".repeat(60)));
  console.log("\n");
}
