import { Options, query } from "@anthropic-ai/claude-agent-sdk";
import {
  red,
  green,
  magentaBright,
  yellow,
  bold,
  cyan,
} from "@visulima/colorize";

export class Agent {
  options: Options;

  constructor(
    options: Options,
    {
      resume = undefined,
      plan = false,
    }: { resume: string | undefined; plan: boolean },
  ) {
    this.options = options;
    this.options.resume = resume;
    if (plan) {
      this.options.permissionMode = "plan";
    } else {
      this.options.permissionMode = "default";
    }
  }

  async run(prompt: string) {
    for await (const message of query({
      prompt: prompt,
      options: this.options,
    })) {
      if (message.type == "assistant") {
        console.log();
        const msg = message.message;
        for (const block of msg.content) {
          if (block.type === "text") {
            console.log(bold(magentaBright("Assistant Response:")));
            console.log(block.text);
          } else if (block.type === "tool_use") {
            console.log(bold(cyan(`Assistant Calling tool ${block.name}`)));
            console.log(bold("Tool input:"));
            console.log(JSON.stringify(block.input, null, 2));
          } else if (block.type === "thinking") {
            console.log(bold(magentaBright("Assistant Thought:")));
            console.log(block.thinking);
          }
        }
      } else if (message.type === "user") {
        console.log();
        const msg = message.message;
        for (const block of msg.content) {
          if (typeof block === "string") {
            console.log(bold("User input:"));
            console.log(block);
          } else {
            if (block.type == "text") {
              console.log(bold("User input:"));
              console.log(block.text);
            } else if (block.type == "tool_result") {
              console.log(
                bold(yellow(`Result for tool: ${block.tool_use_id}`)),
              );
              if (block.content) {
                for (const b of block.content) {
                  if (typeof b === "string") {
                    console.log(b);
                  } else if (b.type == "text") {
                    console.log(b.text);
                  }
                }
              }
            }
          }
        }
      } else if (message.type == "system") {
        if (message.subtype == "init") {
          console.log(bold(`Starting session: ${message.session_id}`));
        } else if (message.subtype == "hook_response") {
          console.log(
            `Hook reponse by ${message.hook_name} for ${message.hook_event}:`,
          );
          console.log("STDOUT:");
          console.log(message.stdout);
          console.log("STDERR:");
          console.log(message.stderr);
        }
      } else if (message.type == "result") {
        console.log();
        if (message.subtype == "success") {
          console.log(green(bold("Final result:")));
          console.log(message.result);
        } else {
          console.log(
            red(bold("One or more errors occurred during the execution:")),
          );
          for (const err of message.errors) {
            console.log(err);
          }
        }
      }
    }
  }
}
