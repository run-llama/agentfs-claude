import {
  HookCallbackMatcher,
  HookEvent,
  HookJSONOutput,
  Options,
  PostToolUseHookInput,
  PreToolUseHookInput,
} from "@anthropic-ai/claude-agent-sdk";
import * as fs from "fs/promises";
import { fileSystemMCP } from "./mcp";
import { consoleInput } from "./cli";
import { filesToParseExtensions } from "./filesystem";

async function denyFileSystemToolsHook(
  // eslint-disable-next-line
  _input: PreToolUseHookInput,
  // eslint-disable-next-line
  _toolUseId: string | undefined,
  // eslint-disable-next-line
  _options: { signal: AbortSignal },
): Promise<HookJSONOutput> {
  return {
    async: true,
    hookSpecificOutput: {
      permissionDecision: "deny",
      permissionDecisionReason:
        "You cannot use standard file system tools, you should use the ones from the filesystem MCP server.",
      hookEventName: "PreToolUse",
    },
  };
}

async function askToPersistFileWrite(
  input: PostToolUseHookInput,
  // eslint-disable-next-line
  _toolUseId: string | undefined,
  // eslint-disable-next-line
  _options: { signal: AbortSignal },
): Promise<HookJSONOutput> {
  const toolInput = input.tool_input;
  let filePath = "";
  let fileContent = "";
  if (typeof toolInput === "object" && toolInput != null) {
    if ("filePath" in toolInput) {
      filePath = toolInput.filePath as string;
    }
    if ("fileContent" in toolInput) {
      fileContent = toolInput.fileContent as string;
    }
  }
  if (
    filePath != "" &&
    fileContent != "" &&
    !filesToParseExtensions.includes("." + filePath.split(".").at(1))
  ) {
    console.log(
      `Would you allow Claude to write ${filePath} with the above content in your real file-system? [y/n]`,
    );
    const answer = await consoleInput("Your answer: ");
    if (["y", "yes", "yse"].includes(answer.toLowerCase().trim())) {
      await fs.writeFile(filePath, fileContent);
      return { async: true, continue: true } as HookJSONOutput;
    } else {
      return { async: true, continue: true } as HookJSONOutput;
    }
  }
  return { async: true, continue: true } as HookJSONOutput;
}

async function askToPersistFileEdit(
  input: PostToolUseHookInput,
  // eslint-disable-next-line
  _toolUseId: string | undefined,
  // eslint-disable-next-line
  _options: { signal: AbortSignal },
): Promise<HookJSONOutput> {
  const toolInput = input.tool_input;
  let filePath = "";
  let oldString = "";
  let newString = "";
  if (typeof toolInput === "object" && toolInput != null) {
    if ("filePath" in toolInput) {
      filePath = toolInput.filePath as string;
    }
    if ("oldString" in toolInput) {
      oldString = toolInput.oldString as string;
    }
    if ("newString" in toolInput) {
      newString = toolInput.newString as string;
    }
  }
  if (
    filePath != "" &&
    oldString != "" &&
    newString != "" &&
    !filesToParseExtensions.includes("." + filePath.split(".").at(1))
  ) {
    console.log(
      `Would you allow Claude to edit ${filePath} with the above content in your real file-system? [y/n]`,
    );
    const answer = await consoleInput("Your answer: ");
    if (["y", "yes", "yse"].includes(answer.toLowerCase().trim())) {
      const content = await fs.readFile(filePath, { encoding: "utf-8" });
      const editedContent = content.replace(oldString, newString);
      await fs.writeFile(filePath, editedContent);
      return { async: true, continue: true } as HookJSONOutput;
    } else {
      return { async: true, continue: true } as HookJSONOutput;
    }
  }
  return { async: true, continue: true } as HookJSONOutput;
}

const hooks: Partial<Record<HookEvent, HookCallbackMatcher[]>> = {
  PreToolUse: [
    {
      matcher: "Read|Write|Edit|Glob",
      hooks: [denyFileSystemToolsHook],
    } as HookCallbackMatcher,
  ],
  PostToolUse: [
    {
      matcher: "mcp__filesystem__write_file",
      hooks: [askToPersistFileWrite],
    } as HookCallbackMatcher,
    {
      matcher: "mcp__filesystem__edit_file",
      hooks: [askToPersistFileEdit],
    } as HookCallbackMatcher,
  ],
};

export const systemPrompt = `
You are an expert programmer whose task is to assist the user implement their requsts within the current working directory.

In order to perform file system operations, you MUST NOT USE the built-in tools you have (Read, Write, Glob, Edit): instead, you MUST USE the 'filesystem' MCP server, wich provides the following tools:

- 'read_file': read a file, providing its path
- 'write_file': write a file, providing its path and content
- 'edit_file': edit a file, providing the old string and the new string to replace the old one with
- 'list_files': list all the available files
- 'file_exists': check whether or not a file exists, providing its path

Using these tools, you should be able to provide the user with the assistance that they need.
`;

export const queryOptions: Options = {
  hooks: hooks,
  disallowedTools: ["Read", "Write", "Edit", "Glob"],
  allowedTools: [
    "mcp__filesystem__list_files",
    "mcp__filesystem__write_file",
    "mcp__filesystem__edit_file",
    "mcp__filesystem__read_file",
    "mcp__filesystem__file_exists",
  ],
  permissionMode: "default",
  systemPrompt: systemPrompt,
  mcpServers: {
    filesystem: fileSystemMCP,
  },
};
