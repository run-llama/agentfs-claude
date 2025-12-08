import { AgentFS } from "agentfs-sdk";
import * as z from "zod";
import { type CallToolResult } from "@modelcontextprotocol/sdk/types.js";
import {
  editFile,
  fileExists,
  listFiles,
  readFile,
  writeFile,
} from "./filesystem";
import { tool, createSdkMcpServer } from "@anthropic-ai/claude-agent-sdk";

const readSchemaShape = {
  filePath: z.string().describe("Path of the file to read"),
};

const writeSchemaShape = {
  filePath: z.string().describe("Path of the file to write"),
  fileContent: z.string().describe("Content of the file"),
};

const editSchemaShape = {
  filePath: z.string().describe("Path of the file to write"),
  oldString: z.string().describe("Old string in the file (to replace)"),
  newString: z
    .string()
    .describe("New string with which to replace the old string"),
};

const fileExistsSchemaShape = {
  filePath: z
    .string()
    .describe("Path of the file whose existence has to be checked"),
};

const listFilesSchemaShape = {};

// eslint-disable-next-line
const readSchema = z.object(readSchemaShape);
// eslint-disable-next-line
const writeSchema = z.object(writeSchemaShape);
// eslint-disable-next-line
const editSchema = z.object(editSchemaShape);
// eslint-disable-next-line
const fileExistsSchema = z.object(fileExistsSchemaShape);
// eslint-disable-next-line
const listFilesSchema = z.object(listFilesSchemaShape);

export async function getAgentFS({
  filePath = null,
}: {
  filePath?: string | null;
}): Promise<AgentFS> {
  if (!filePath) {
    filePath = "fs.db";
  }
  const agentfs = await AgentFS.open({ id: "claude-agentfs", path: filePath });
  return agentfs;
}

async function readTool(
  input: z.infer<typeof readSchema>,
): Promise<CallToolResult> {
  const agentfs = await getAgentFS({});
  const content = await readFile(input.filePath, agentfs);
  if (typeof content == "string") {
    return { content: [{ type: "text", text: content }] };
  } else {
    return {
      content: [
        {
          type: "text",
          text: `Could not read ${input.filePath}. Please check that the file exists and submit the request again.`,
        },
      ],
      isError: true,
    };
  }
}

async function fileExistsTool(
  input: z.infer<typeof fileExistsSchema>,
): Promise<CallToolResult> {
  const agentfs = await getAgentFS({});
  const exists = await fileExists(input.filePath, agentfs);
  if (exists) {
    return {
      content: [{ type: "text", text: `File ${input.filePath} exists` }],
    };
  } else {
    return {
      content: [
        { type: "text", text: `File ${input.filePath} does not exist.` },
      ],
    };
  }
}

async function writeFileTool(
  input: z.infer<typeof writeSchema>,
): Promise<CallToolResult> {
  const agentfs = await getAgentFS({});
  const success = await writeFile(input.filePath, input.fileContent, agentfs);
  if (success) {
    return {
      content: [
        {
          type: "text",
          text: `File ${input.filePath} successfully written with content:\n\n'''\n${input.fileContent}\n'''`,
        },
      ],
    };
  } else {
    return {
      content: [
        {
          type: "text",
          text: `There was an error while writing file ${input.filePath}`,
        },
      ],
    };
  }
}

async function editFileTool(
  input: z.infer<typeof editSchema>,
): Promise<CallToolResult> {
  const agentfs = await getAgentFS({});
  const editedContent = await editFile(
    input.filePath,
    input.oldString,
    input.newString,
    agentfs,
  );
  if (typeof editedContent == "string") {
    return {
      content: [
        {
          type: "text",
          text: `Successfully edited ${input.filePath}. New content:\n\n'''\n${editedContent}\n'''`,
        },
      ],
    } as CallToolResult;
  } else {
    return {
      content: [
        {
          type: "text",
          text: `Could not edit ${input.filePath}. Please check that the file exists and submit the request again.`,
        },
      ],
      isError: true,
    };
  }
}

async function listFilesTool(
  input: z.infer<typeof listFilesSchema>,
): Promise<CallToolResult> {
  const agentfs = await getAgentFS(input);
  const files = await listFiles(agentfs);
  if (files != "") {
    return { content: [{ type: "text", text: files }] };
  } else {
    return {
      content: [
        {
          type: "text",
          text: `Could not list files. Please report this failure to the user`,
        },
      ],
      isError: true,
    };
  }
}

const mcpReadTool = tool(
  "read_file",
  "Read a file by passing its path.",
  readSchemaShape,
  readTool,
);

const mcpWriteTool = tool(
  "write_file",
  "Write a file by passing its path and content.",
  writeSchemaShape,
  writeFileTool,
);

const mcpEditTool = tool(
  "edit_file",
  "Edit a file by passing its path, the old string and the new string.",
  editSchemaShape,
  editFileTool,
);

const mcpFileExists = tool(
  "file_exists",
  "Check whether a file exists or not by passing its path.",
  fileExistsSchemaShape,
  fileExistsTool,
);

const mcplistFiles = tool(
  "list_files",
  "List all the available files",
  listFilesSchemaShape,
  listFilesTool,
);

export const fileSystemMCP = createSdkMcpServer({
  name: "filesystem-mcp",
  version: "1.0.0",
  tools: [mcpReadTool, mcpWriteTool, mcplistFiles, mcpEditTool, mcpFileExists],
});
