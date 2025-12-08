import {
  HookCallbackMatcher,
  HookEvent,
  HookJSONOutput,
  Options,
  PreToolUseHookInput,
} from "@anthropic-ai/claude-agent-sdk";
import { fileSystemMCP } from "./mcp";

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

const hooks: Partial<Record<HookEvent, HookCallbackMatcher[]>> = {
  PreToolUse: [
    {
      matcher: "Read|Write|Edit|Glob",
      hooks: [denyFileSystemToolsHook],
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
