import { Codex } from "@openai/codex-sdk";
import type { ThreadEvent, Thread } from "@openai/codex-sdk";
import {
  red,
  green,
  magentaBright,
  yellow,
  bold,
  cyan,
} from "@visulima/colorize";

export const codex = new Codex({});

export async function runCodex(
  prompt: string,
  { resumeSession = undefined }: { resumeSession: string | undefined },
) {
  let thread: Thread | undefined = undefined;
  if (typeof resumeSession == "undefined") {
    thread = codex.startThread({
      skipGitRepoCheck: true,
    });
  } else {
    thread = codex.resumeThread(resumeSession, {
      skipGitRepoCheck: true,
    });
  }
  const { events } = await thread.runStreamed(prompt);

  for await (const event of events) {
    switch (event.type) {
      case "thread.started":
        console.log(`Started session with ID: ${bold(event.thread_id)}`);
        break;
      case "item.started":
        await handleItemStart(event);
        break;
      case "item.updated":
        await handleItemUpdated(event);
        break;
      case "item.completed":
        await handleItemCompleted(event);
        break;
      case "turn.completed":
        await handleTurnCompletion(event);
        break;
      case "error":
        await handleError(event);
        break;
    }
  }
}

async function handleItemStart(event: ThreadEvent) {
  if (event.type == "item.started") {
    if (event.item.type == "agent_message") {
      console.log(bold(magentaBright("Assistant started responding...")));
      console.log(event.item.text);
    } else if (event.item.type == "reasoning") {
      console.log(bold(magentaBright("Assistant started thinking...")));
      console.log(event.item.text);
    } else if (event.item.type == "mcp_tool_call") {
      console.log(
        bold(
          yellow(
            `Assistant started calling MCP tool ${event.item.tool} with input ${JSON.stringify(event.item.arguments)}`,
          ),
        ),
      );
      if (event.item.error) {
        console.log(
          red(
            bold(
              `An error occurred while calling the tool: ${event.item.error.message}`,
            ),
          ),
        );
      } else {
        if (event.item.result) {
          let finalResult = "";
          for (const block of event.item.result.content) {
            if (block.type == "text") {
              finalResult += block.text + "\n";
            }
          }
          console.log(`${green(bold("Tool result"))}: ${finalResult}`);
        }
      }
    } else if (event.item.type == "todo_list") {
      console.log(
        bold(green("Assistant started to produce a TODO list with items:")),
      );
      let c = 0;
      for (const i of event.item.items) {
        c += 1;
        console.log(
          `TODO item ${c}: ${i.text} (${i.completed ? "completed" : "not completed"})`,
        );
      }
    } else if (event.item.type == "web_search") {
      console.log(
        `Assistant started searching the web with query: ${event.item.query}`,
      );
    } else if (event.item.type == "command_execution") {
      console.log(
        `Assistant started to execute command: ${event.item.command}`,
      );
    } else if (event.item.type == "file_change") {
      console.log(
        bold(red("WARNING! The assistant is starting to change files:")),
      );
      for (const change of event.item.changes) {
        console.log(
          `The assistant would like to apply ${change.kind} to ${change.path}`,
        );
      }
    } else {
      console.log(bold(red(`An error occurred: ${event.item.message}`)));
    }
  }
}

async function handleItemUpdated(event: ThreadEvent) {
  if (event.type == "item.updated") {
    if (event.item.type == "agent_message") {
      console.log(bold(magentaBright("Assistant updated its response...")));
      console.log(event.item.text);
    } else if (event.item.type == "reasoning") {
      console.log(bold(magentaBright("Assistant updated its thoughts...")));
      console.log(event.item.text);
    } else if (event.item.type == "mcp_tool_call") {
      console.log(
        bold(
          yellow(
            `Assistant updated its call to MCP tool ${event.item.tool} with input ${JSON.stringify(event.item.arguments)}`,
          ),
        ),
      );
      if (event.item.error) {
        console.log(
          red(
            bold(
              `An error occurred while calling the tool: ${event.item.error.message}`,
            ),
          ),
        );
      } else {
        if (event.item.result) {
          let finalResult = "";
          for (const block of event.item.result.content) {
            if (block.type == "text") {
              finalResult += block.text + "\n";
            }
          }
          console.log(`${green(bold("Tool result"))}: ${finalResult}`);
        }
      }
    } else if (event.item.type == "todo_list") {
      console.log(bold(green("Assistant updated its TODO list with items:")));
      let c = 0;
      for (const i of event.item.items) {
        c += 1;
        console.log(
          `TODO item ${c}: ${i.text} (${i.completed ? "completed" : "not completed"})`,
        );
      }
    } else if (event.item.type == "web_search") {
      console.log(
        `Assistant updated its web search with query: ${event.item.query}`,
      );
    } else if (event.item.type == "command_execution") {
      console.log(`Assistant updated command execution: ${event.item.command}`);
    } else if (event.item.type == "file_change") {
      console.log(
        bold(red("WARNING! The assistant is starting to change files:")),
      );
      for (const change of event.item.changes) {
        console.log(
          `The assistant is updating the change it would like to apply (${change.kind}) to ${change.path}`,
        );
      }
    } else {
      console.log(bold(red(`An error occurred: ${event.item.message}`)));
    }
  }
}

async function handleItemCompleted(event: ThreadEvent) {
  if (event.type == "item.completed") {
    if (event.item.type == "agent_message") {
      console.log(bold(magentaBright("Assistant completed its response:")));
      console.log(event.item.text);
    } else if (event.item.type == "reasoning") {
      console.log(bold(magentaBright("Assistant completed its thoughts:")));
      console.log(event.item.text);
    } else if (event.item.type == "mcp_tool_call") {
      console.log(
        bold(
          yellow(
            `Assistant completed its call to MCP tool ${event.item.tool} with input ${JSON.stringify(event.item.arguments)}`,
          ),
        ),
      );
      if (event.item.error) {
        console.log(
          red(
            bold(
              `An error occurred while calling the tool: ${event.item.error.message}`,
            ),
          ),
        );
      } else {
        if (event.item.result) {
          let finalResult = "";
          for (const block of event.item.result.content) {
            if (block.type == "text") {
              finalResult += block.text + "\n";
            }
          }
          console.log(`${green(bold("Tool result"))}: ${finalResult}`);
        }
      }
    } else if (event.item.type == "todo_list") {
      console.log(bold(green("Assistant completed its TODO list with items:")));
      let c = 0;
      for (const i of event.item.items) {
        c += 1;
        console.log(
          `TODO item ${c}: ${i.text} (${i.completed ? "completed" : "not completed"})`,
        );
      }
    } else if (event.item.type == "web_search") {
      console.log(
        `Assistant updated its web search with query: ${event.item.query}`,
      );
    } else if (event.item.type == "command_execution") {
      console.log(`Assistant updated command execution: ${event.item.command}`);
    } else if (event.item.type == "file_change") {
      console.log(
        bold(red("ERROR! The assistant has completed its file changes:")),
      );
      for (const change of event.item.changes) {
        console.log(
          `The assistant has completed the change it applied to ${change.path} (${change.kind})`,
        );
      }
    } else {
      console.log(bold(red(`An error occurred: ${event.item.message}`)));
    }
  }
}

async function handleTurnCompletion(event: ThreadEvent) {
  if (event.type == "turn.completed") {
    console.log(cyan(bold("Turn completed, usage:")));
    console.log(`Input tokens: ${event.usage.input_tokens}`);
    console.log(`Cached input tokens: ${event.usage.cached_input_tokens}`);
    console.log(`Output tokens: ${event.usage.output_tokens}`);
  }
}

async function handleError(event: ThreadEvent) {
  if (event.type == "error") {
    console.log(bold(red(`An error occurred: ${event.message}`)));
  }
}
